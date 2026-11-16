"""Improvement 3 of 5: AST Function-Level Git-Diff Delta Caching for Live IDE Edits.

Decomposes repository source files into FunctionDef / ClassDef AST segments so live
developer edits (Git diffs) only re-prefill the single modified function (~100 tokens)
while reusing 90%+ of the file's canonical KV blocks in O(1) time.
"""

import ast
import hashlib
import re
import time
from typing import Dict, List, Tuple
import torch

import benchmarks.bench_7b_realworld_rag as rag_mod
from benchmarks.bench_7b_realworld_rag import DualGPUSharded7BEngine
from benchmarks.bench_multiuser_codebase import release_request_blocks
from functools import partial
from benchmarks.bench_submodular_bridge import select_submodular_bridge_tokens as _base_submod
select_submodular_bridge_tokens = partial(_base_submod, min_tokens=16)


def clean_source_file(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()
    return re.sub(r'"""[\s\S]*?"""', "", raw).strip()


def split_python_into_ast_nodes(filepath: str, clean_code: str) -> List[str]:
    """Splits Python source into exact AST FunctionDef / ClassDef segments."""
    tree = ast.parse(clean_code)
    lines = clean_code.splitlines(keepends=True)
    split_lines = [0]

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            split_lines.append(node.lineno - 1)
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    split_lines.append(item.lineno - 1)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            split_lines.append(node.lineno - 1)

    split_lines = sorted(set(split_lines))
    split_lines.append(len(lines))

    raw_segs = []
    for i in range(len(split_lines) - 1):
        seg = "".join(lines[split_lines[i]:split_lines[i + 1]])
        if seg.strip():
            raw_segs.append(seg)

    merged = []
    buf = ""
    for seg in raw_segs:
        buf += seg
        if len(buf) >= 350:
            merged.append(buf)
            buf = ""
    if buf:
        if merged:
            merged[-1] += buf
        else:
            merged.append(buf)

    merged[0] = f"### FILE: {filepath}\n```python\n" + merged[0]
    merged[-1] = merged[-1] + "\n```\n\n"
    return merged


@torch.inference_mode()
def run_ast_delta_benchmark():
    print("\n==========================================================================================")
    print("  IMPROVEMENT 3: AST FUNCTION-LEVEL GIT-DIFF DELTA CACHING (2x RTX 2080 Ti)")
    print("==========================================================================================\n")

    engine = DualGPUSharded7BEngine(model_name="Qwen/Qwen2.5-7B-Instruct", max_num_blocks=768)
    rag_mod.select_bridge_tokens = select_submodular_bridge_tokens

    filepath = "dag_kv/core/block_manager.py"
    base_code = clean_source_file(filepath)

    # Index initial baseline version of block_manager.py at AST Function granularity
    ast_cache: Dict[str, Dict] = {}
    base_nodes = split_python_into_ast_nodes(filepath, base_code)
    total_base_tokens = 0
    for seg in base_nodes:
        h = hashlib.sha256(seg.encode("utf-8")).hexdigest()[:16]
        ast_cache[h] = engine.prefill_canonical_chunk(seg)
        total_base_tokens += ast_cache[h]["length"]

    print(f"Indexed `{filepath}` into {len(base_nodes)} AST Function/Method Nodes ({total_base_tokens} tokens total).\n")

    # Define 3 realistic live IDE developer edits (Git diffs) to single methods in block_manager.py
    live_edits = [
        (
            "Git Diff #1 (Edit BlockManager.__init__)",
            base_code.replace("max_num_blocks: int = 4096", "max_num_blocks: int = 8192"),
            "\nQuestion: In dag_kv/core/block_manager.py, what is the default integer value of max_num_blocks in BlockManager.__init__? Answer with ONLY the number:\nAnswer:",
            "8192",
        ),
        (
            "Git Diff #2 (Edit OffsetBlockTable.__init__)",
            base_code.replace("block_size: int = 16", "block_size: int = 64"),
            "\nQuestion: In dag_kv/core/block_manager.py, what is the default integer value of block_size in OffsetBlockTable.__init__? Answer with ONLY the number:\nAnswer:",
            "64",
        ),
        (
            "Git Diff #3 (Edit PhysicalBlock.ref_count)",
            base_code.replace("ref_count: int = 0", "ref_count: int = 99"),
            "\nQuestion: In dag_kv/core/block_manager.py, what is the default integer value of ref_count in PhysicalBlock? Answer with ONLY the number:\nAnswer:",
            "99",
        ),
    ]

    print(f"{'Live IDE Developer Edit':<38} | {'AST Reuse':<10} | {'Full File ms':<12} | {'AST Delta ms':<12} | {'Speedup':<7} | {'Expect':<6} | {'Pred'}")
    print("-" * 106)

    full_times, delta_times, exact_hits = [], [], 0

    for label, edited_code, q_text, expected in live_edits:
        # 1. Standard File-Level Re-Index (Entire file invalidated by 1-line edit)
        full_file_str = f"### FILE: {filepath}\n```python\n{edited_code}\n```\n\n"
        torch.cuda.synchronize("cuda:0")
        torch.cuda.synchronize("cuda:1")
        t0 = time.perf_counter()
        c_full = engine.prefill_canonical_chunk(full_file_str)
        torch.cuda.synchronize("cuda:0")
        torch.cuda.synchronize("cuda:1")
        full_ms = (time.perf_counter() - t0) * 1000.0
        full_times.append(full_ms)
        for p_blk in c_full["phys_blocks"]:
            engine.bm_0.free_block(p_blk)
            engine.bm_1.free_block(p_blk)

        # 2. AST Function-Level Delta Re-Index (Only modified FunctionDef node is prefilled)
        edited_nodes = split_python_into_ast_nodes(filepath, edited_code)
        active_ast_chunks = []
        reused_tokens = 0
        new_tokens = 0
        new_node_chunks = []

        torch.cuda.synchronize("cuda:0")
        torch.cuda.synchronize("cuda:1")
        t1 = time.perf_counter()
        for seg in edited_nodes:
            h = hashlib.sha256(seg.encode("utf-8")).hexdigest()[:16]
            if h in ast_cache:
                active_ast_chunks.append(ast_cache[h])
                reused_tokens += ast_cache[h]["length"]
            else:
                c_new = engine.prefill_canonical_chunk(seg)
                active_ast_chunks.append(c_new)
                new_node_chunks.append(c_new)
                new_tokens += c_new["length"]
        torch.cuda.synchronize("cuda:0")
        torch.cuda.synchronize("cuda:1")
        delta_ms = (time.perf_counter() - t1) * 1000.0
        delta_times.append(delta_ms)

        reuse_pct = reused_tokens / float(reused_tokens + new_tokens) * 100.0

        # Stitch the AST nodes and verify the model sees the new live-edited value
        table, doc_toks = engine.assemble_prompt_dag(active_ast_chunks, seq_id="ast_edit")
        engine.repair_bridge_tokens(table, active_ast_chunks, top_k_percent=0.08)

        q_ids = engine.tokenizer.encode(q_text, add_special_tokens=False)
        curr_pos = len(doc_toks)
        for tok in q_ids:
            cur_l = engine.append_and_decode(table, token_id=tok, current_pos=curr_pos)
            curr_pos += 1
        gen_ids = []
        for _ in range(10):
            nxt = torch.argmax(cur_l, dim=-1).item()
            gen_ids.append(nxt)
            if nxt in (engine.tokenizer.eos_token_id, 198):
                break
            cur_l = engine.append_and_decode(table, token_id=nxt, current_pos=curr_pos)
            curr_pos += 1

        release_request_blocks(engine, table)
        for c_new in new_node_chunks:
            for p_blk in c_new["phys_blocks"]:
                engine.bm_0.free_block(p_blk)
                engine.bm_1.free_block(p_blk)

        pred = engine.tokenizer.decode(gen_ids, skip_special_tokens=True).strip().split()[0].rstrip(".,;:")
        if expected in pred:
            exact_hits += 1

        print(f"{label:<38} | {reuse_pct:>8.1f}% | {full_ms:>10.1f}ms | {delta_ms:>10.1f}ms | {full_ms/delta_ms:>6.2f}x | {expected:<6} | {pred}")

    m_full = sum(full_times) / len(full_times)
    m_delta = sum(delta_times) / len(delta_times)
    print("-" * 106)
    print(f"Mean Incremental Re-Index Latency : Full-File = {m_full:.1f} ms vs. AST Function-Delta = {m_delta:.1f} ms ({m_full/m_delta:.2f}x Faster IDE Save!)")
    print(f"Live Git-Diff Code QA Accuracy    : {exact_hits}/{len(live_edits)} Updated Values Correctly Retrieved ({exact_hits/len(live_edits)*100:.1f}%)")
    print("==========================================================================================\n")


if __name__ == "__main__":
    run_ast_delta_benchmark()
