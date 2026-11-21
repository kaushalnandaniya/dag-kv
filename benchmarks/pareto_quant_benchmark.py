#!/usr/bin/env python3
"""Real-weight Pareto benchmark for DAG-KV (FP16 vs INT8 vs Group-32 INT4).

Evaluates KV VRAM footprint, decode throughput, warm-permutation latency,
multi-chunk QA exact-match accuracy, and logit KL-divergence vs. native HF FP16.
"""

import argparse
import gc
import json
import os
import time
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from dag_kv.models import AutoDAGKVEngine

CHUNKS = {
    "billing": (
        "### FILE: billing.py\n"
        "DEFAULT_TAX_RATE = 0.18\n"
        "RETRY_LIMIT = 5\n"
        "CURRENCY_CODE = 'INR'\n"
        "def compute_total(subtotal):\n"
        "    return subtotal * (1.0 + DEFAULT_TAX_RATE)\n\n"
    ),
    "auth": (
        "### FILE: auth.py\n"
        "TOKEN_TTL_SECONDS = 3600\n"
        "SIGNING_ALGO = 'HS256'\n"
        "MAX_LOGIN_ATTEMPTS = 7\n"
        "def is_expired(age_sec):\n"
        "    return age_sec > TOKEN_TTL_SECONDS\n\n"
    ),
    "cache": (
        "### FILE: cache_policy.py\n"
        "EVICTION_POLICY = 'LRU'\n"
        "MAX_CACHE_ENTRIES = 4096\n"
        "SHARD_COUNT = 12\n"
        "def shard_for_key(key_hash):\n"
        "    return key_hash % SHARD_COUNT\n\n"
    ),
    "limiter": (
        "### FILE: rate_limiter.py\n"
        "REQUESTS_PER_MINUTE = 240\n"
        "BURST_MULTIPLIER = 2.5\n"
        "HEADER_NAME = 'X-RateLimit-Remaining'\n"
        "def burst_cap():\n"
        "    return int(REQUESTS_PER_MINUTE * BURST_MULTIPLIER)\n\n"
    ),
}

# Queries span both forward [billing, auth, cache, limiter] and permuted [limiter, cache, billing, auth] DAGs
EVAL_CASES = [
    {
        "name": "fwd_tax_and_ttl",
        "order": ["billing", "auth", "cache", "limiter"],
        "question": "Question: What are the exact values of DEFAULT_TAX_RATE and TOKEN_TTL_SECONDS? Answer concisely.",
        "expected": ["0.18", "3600"],
    },
    {
        "name": "fwd_cache_and_burst",
        "order": ["billing", "auth", "cache", "limiter"],
        "question": "Question: What are MAX_CACHE_ENTRIES and REQUESTS_PER_MINUTE? Answer concisely.",
        "expected": ["4096", "240"],
    },
    {
        "name": "perm_algo_and_retry",
        "order": ["limiter", "cache", "billing", "auth"],
        "question": "Question: What are SIGNING_ALGO and RETRY_LIMIT? Answer concisely.",
        "expected": ["HS256", "5"],
    },
    {
        "name": "perm_policy_and_header",
        "order": ["limiter", "cache", "billing", "auth"],
        "question": "Question: What are EVICTION_POLICY and HEADER_NAME? Answer concisely.",
        "expected": ["LRU", "X-RateLimit-Remaining"],
    },
    {
        "name": "perm_shards_and_currency",
        "order": ["auth", "limiter", "billing", "cache"],
        "question": "Question: What are SHARD_COUNT and CURRENCY_CODE? Answer concisely.",
        "expected": ["12", "INR"],
    },
    {
        "name": "perm_attempts_and_mult",
        "order": ["cache", "billing", "limiter", "auth"],
        "question": "Question: What are MAX_LOGIN_ATTEMPTS and BURST_MULTIPLIER? Answer concisely.",
        "expected": ["7", "2.5"],
    },
]


def get_turn_suffix(tokenizer) -> str:
    """Derives the model-native assistant turn suffix via chat_template."""
    sentinel = "<<<DAGKV_SENTINEL>>>"
    try:
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": sentinel}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if sentinel in rendered:
            return rendered.split(sentinel, 1)[1]
    except Exception:
        pass
    return "\nAnswer: "


def get_stop_token_ids(tokenizer) -> set:
    stops = set()
    if tokenizer.eos_token_id is not None:
        stops.add(int(tokenizer.eos_token_id))
    for tok_str in ("<|im_end|>", "<|eot_id|>", "<|end_of_text|>", "<end_of_turn>"):
        tid = tokenizer.convert_tokens_to_ids(tok_str)
        if isinstance(tid, int) and tid >= 0 and tid != tokenizer.unk_token_id:
            stops.add(tid)
    return stops


def load_hf_model(model_name: str, device: str):
    """Loads from local cache first, falling back to Hugging Face Hub if missing."""
    try:
        tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16, device_map=device, local_files_only=True
        ).eval()
    except OSError:
        tok = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16, device_map=device
        ).eval()
    return model, tok


def measure_kv_pool_bytes(eng) -> int:
    bm = eng.block_manager
    tensors = [
        bm.k_pool,
        bm.v_pool,
        getattr(bm, "k_scale_pool", None),
        getattr(bm, "v_scale_pool", None),
    ]
    return sum(t.numel() * t.element_size() for t in tensors if t is not None)


@torch.inference_mode()
def compute_hf_reference_logits(model, full_token_ids: List[int], device: str) -> torch.Tensor:
    inp = torch.tensor([full_token_ids], dtype=torch.long, device=device)
    out = model(input_ids=inp, use_cache=False)
    return out.logits[0, -1, :].float()


def benchmark_model(
    model_name: str,
    device: str,
    quant_modes: List[str],
    max_num_blocks: int,
    block_size: int,
    max_new_tokens: int,
) -> List[Dict]:
    print(f"\n{'=' * 78}\nLoading {model_name} on {device}...\n{'=' * 78}")
    model, tok = load_hf_model(model_name, device)
    turn_suffix = get_turn_suffix(tok)
    stop_ids = get_stop_token_ids(tok)
    hf_ref_cache: Dict[Tuple[int, ...], torch.Tensor] = {}

    results = []
    fp16_pool_bytes = None

    for quant in quant_modes:
        torch.cuda.empty_cache()
        eng = AutoDAGKVEngine.from_model(
            model=model,
            tokenizer=tok,
            device=device,
            dtype=torch.float16,
            quantization=quant,
            max_num_blocks=max_num_blocks,
            block_size=block_size,
        )

        pool_bytes = measure_kv_pool_bytes(eng)
        if quant == "none":
            fp16_pool_bytes = pool_bytes
        bytes_per_tok = pool_bytes / (max_num_blocks * block_size)
        compression_ratio = (fp16_pool_bytes / pool_bytes) if fp16_pool_bytes else 1.0

        # Prefill canonical chunks once (shared across all forward & permuted queries)
        torch.cuda.synchronize(device)
        t_prefill0 = time.perf_counter()
        canonical_map = {k: eng.prefill_canonical_chunk(v) for k, v in CHUNKS.items()}
        torch.cuda.synchronize(device)
        cold_prefill_ms = (time.perf_counter() - t_prefill0) * 1000.0

        # Warmup Triton kernel compilation (2 decode steps) before timing
        warm_table, warm_doc = eng.assemble_prompt_dag([canonical_map["billing"]], seq_id="warmup")
        w_pos = len(warm_doc)
        for _ in range(2):
            _ = eng.append_and_decode(warm_table, token_id=10, current_pos=w_pos)
            w_pos += 1
        eng.block_manager.free_sequence(warm_table)

        exact_hits = 0
        kl_list = []
        cos_list = []
        top1_matches = 0
        decode_step_times_ms = []
        stitch_repair_times_ms = []
        case_details = []

        for idx, case in enumerate(EVAL_CASES):
            ordered_chunks = [canonical_map[k] for k in case["order"]]

            torch.cuda.synchronize(device)
            t_st0 = time.perf_counter()
            table, doc_toks = eng.assemble_prompt_dag(ordered_chunks, seq_id=f"{quant}_{idx}")
            rep = eng.repair_bridge_tokens(table, ordered_chunks)
            torch.cuda.synchronize(device)
            stitch_repair_times_ms.append((time.perf_counter() - t_st0) * 1000.0)

            query_text = "\n" + case["question"] + turn_suffix
            q_ids = tok.encode(query_text, add_special_tokens=False)

            pos = len(doc_toks)
            last_logits = None
            for q_tok in q_ids:
                last_logits = eng.append_and_decode(table, token_id=q_tok, current_pos=pos)
                pos += 1

            # Compare first-token distribution against native HF FP16 on exact token sequence
            full_key = tuple(doc_toks + q_ids)
            if full_key not in hf_ref_cache:
                hf_ref_cache[full_key] = compute_hf_reference_logits(model, list(full_key), device)
            hf_logits = hf_ref_cache[full_key]
            dag_logits = last_logits.squeeze().float()

            p_hf = F.softmax(hf_logits, dim=-1)
            log_p_hf = F.log_softmax(hf_logits, dim=-1)
            log_p_dag = F.log_softmax(dag_logits, dim=-1)
            kl = float(torch.sum(p_hf * (log_p_hf - log_p_dag)).item())
            cos = float(F.cosine_similarity(hf_logits, dag_logits, dim=0).item())
            top1_ok = int(torch.argmax(hf_logits).item() == torch.argmax(dag_logits).item())
            kl_list.append(kl)
            cos_list.append(cos)
            top1_matches += top1_ok

            # Greedy decode & measure per-token decode latency
            gen_ids = [int(torch.argmax(dag_logits).item())]
            torch.cuda.synchronize(device)
            t_dec0 = time.perf_counter()
            timed_steps = 0
            for _ in range(max_new_tokens - 1):
                if gen_ids[-1] in stop_ids:
                    break
                logits = eng.append_and_decode(table, token_id=gen_ids[-1], current_pos=pos)
                pos += 1
                timed_steps += 1
                gen_ids.append(int(torch.argmax(logits.squeeze(), dim=-1).item()))
            torch.cuda.synchronize(device)
            if timed_steps > 0:
                decode_step_times_ms.append(((time.perf_counter() - t_dec0) * 1000.0) / timed_steps)

            decoded = tok.decode(gen_ids, skip_special_tokens=True).strip()
            em_ok = all(exp.lower() in decoded.lower() for exp in case["expected"])
            exact_hits += int(em_ok)

            case_details.append({
                "case": case["name"],
                "repaired": rep.get("num_repaired", 0),
                "em": em_ok,
                "kl": round(kl, 4),
                "logit_cos": round(cos, 4),
                "top1_match": bool(top1_ok),
                "output": decoded[:140],
            })
            eng.block_manager.free_sequence(table)

        avg_ms_tok = sum(decode_step_times_ms) / max(1, len(decode_step_times_ms))
        tok_per_sec = 1000.0 / avg_ms_tok if avg_ms_tok > 0 else 0.0
        avg_stitch_ms = sum(stitch_repair_times_ms) / max(1, len(stitch_repair_times_ms))
        em_pct = 100.0 * exact_hits / len(EVAL_CASES)
        top1_pct = 100.0 * top1_matches / len(EVAL_CASES)
        mean_kl = sum(kl_list) / len(kl_list)
        mean_cos = sum(cos_list) / len(cos_list)

        row = {
            "model": model_name,
            "quantization": quant,
            "kv_pool_mib_8k": round(pool_bytes / (1024**2), 2),
            "bytes_per_tok": round(bytes_per_tok, 1),
            "compression_x": round(compression_ratio, 2),
            "cold_prefill_ms": round(cold_prefill_ms, 1),
            "warm_stitch_repair_ms": round(avg_stitch_ms, 1),
            "decode_ms_per_tok": round(avg_ms_tok, 2),
            "decode_tok_per_sec": round(tok_per_sec, 1),
            "qa_em_pct": round(em_pct, 1),
            "qa_hits": f"{exact_hits}/{len(EVAL_CASES)}",
            "top1_match_pct": round(top1_pct, 1),
            "mean_kl_vs_hf": round(mean_kl, 4),
            "mean_logit_cos": round(mean_cos, 4),
            "cases": case_details,
        }
        results.append(row)
        print(
            f"[{model_name.split('/')[-1]} | {quant:<4}] "
            f"KV(8K)={row['kv_pool_mib_8k']:6.1f} MiB ({row['compression_x']:.2f}x) | "
            f"decode={row['decode_ms_per_tok']:5.1f} ms/tok ({row['decode_tok_per_sec']:4.1f} tok/s) | "
            f"EM={row['qa_hits']} ({row['qa_em_pct']:5.1f}%) | "
            f"KL={row['mean_kl_vs_hf']:.4f} | cos={row['mean_logit_cos']:.4f}"
        )

        del eng
        gc.collect()
        torch.cuda.empty_cache()

    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    return results


def print_markdown_table(all_rows: List[Dict]):
    print("\n### Real-Weight DAG-KV Pareto Benchmark (RTX 2080 Ti)\n")
    header = (
        "| Model | Quant | KV Pool (8K slots) | Compression | Stitch+Repair | "
        "Decode (ms/tok) | Throughput | Multi-Chunk EM | Top-1 vs HF | Mean KL |"
    )
    sep = "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    print(header)
    print(sep)
    for r in all_rows:
        short_m = r["model"].split("/")[-1]
        q_label = {"none": "FP16", "int8": "INT8", "k8v4": "K8V4 (g32)", "int4": "INT4 (K8g/V32g)"}[r["quantization"]]
        print(
            f"| {short_m} | {q_label} | {r['kv_pool_mib_8k']:.1f} MiB | {r['compression_x']:.2f}x | "
            f"{r['warm_stitch_repair_ms']:.1f} ms | {r['decode_ms_per_tok']:.1f} ms | "
            f"{r['decode_tok_per_sec']:.1f} tok/s | {r['qa_hits']} ({r['qa_em_pct']:.0f}%) | "
            f"{r['top1_match_pct']:.0f}% | {r['mean_kl_vs_hf']:.4f} |"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["Qwen/Qwen2.5-3B-Instruct", "meta-llama/Llama-3.2-3B-Instruct"],
    )
    parser.add_argument("--quant", nargs="+", default=["none", "int8", "k8v4", "int4"])
    parser.add_argument("--max-blocks", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--out", default="benchmarks/pareto_results.json")
    args = parser.parse_args()

    if args.device.startswith("cuda"):
        torch.cuda.set_device(args.device)

    all_rows = []
    for m in args.models:
        try:
            all_rows.extend(
                benchmark_model(
                    model_name=m,
                    device=args.device,
                    quant_modes=args.quant,
                    max_num_blocks=args.max_blocks,
                    block_size=args.block_size,
                    max_new_tokens=args.max_new_tokens,
                )
            )
        except Exception as exc:
            print(f"[ERROR] Skipping {m}: {type(exc).__name__}: {exc}")

    print_markdown_table(all_rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\n[SAVED] Full benchmark results written to {args.out}")


if __name__ == "__main__":
    main()
