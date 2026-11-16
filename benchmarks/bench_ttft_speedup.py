"""Benchmark: Time-To-First-Token (TTFT) Speedup across Context Lengths (1K to 8K/16K).

Compares:
1. Full Dense Prefill Baseline: Standard O(N^2) quadratic prefill of permuted chunks.
2. DAG-KV Stitched Cache: Zero-copy canonical KV reuse + 8% Bridge Token Repair.

Measures:
- TTFT latency in milliseconds.
- Speedup factor (T_dense / T_dag).
- Effective prefill throughput (tokens/sec).
"""

import argparse
import time
from typing import Dict, List
import torch
from transformers import AutoTokenizer

from dag_kv.models.qwen2_dag_patch import Qwen2DAGKVEngine


def run_ttft_benchmark(
    lengths: List[int] = [1024, 2048, 4096, 8192],
    num_chunks: int = 4,
    device: str = "cuda:0",
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
) -> List[Dict]:
    print(f"\n=======================================================")
    print(f"  DAG-KV TTFT SPEEDUP BENCHMARK (2x RTX 2080 Ti sm_75)")
    print(f"  Model: {model_name} (FP16)")
    print(f"=======================================================\n")

    engine = Qwen2DAGKVEngine.from_pretrained(model_name, device=device)
    tokenizer = engine.tokenizer

    results = []

    # Sample text corpus
    base_text = (
        "In artificial intelligence and machine learning, key-value caching is a critical "
        "optimization for autoregressive large language model inference. When processing "
        "multi-turn dialogues or retrieval-augmented generation pipelines, identical context "
        "chunks frequently reappear across requests in dynamically permuted orderings. "
        "Standard prefix caching architectures fail to recognize chunks unless they share "
        "an exact common prefix starting at position zero. DAG-KV decouples token position "
        "from cached key vectors using Query-Side Adjoint RoPE in SO(2) and selectively repairs "
        "cross-chunk causal attention via Log-Sum-Exp perturbation bounds. "
    )

    for total_len in lengths:
        print(f"\nEvaluating Context Length: {total_len} tokens ({num_chunks} chunks)...")
        chunk_len = total_len // num_chunks

        # Generate token chunks
        raw_tokens = tokenizer.encode(base_text * 50, add_special_tokens=False)
        chunks_tokens = []
        for i in range(num_chunks):
            start = (i * chunk_len) % (len(raw_tokens) - chunk_len)
            chunks_tokens.append(raw_tokens[start : start + chunk_len])

        # 1. Warm-up and Canonical Prefill into Pool
        print(f"  [1/3] Populating Canonical Pool ({num_chunks} chunks x {chunk_len} tokens)...")
        canon_chunks = []
        for i, ct in enumerate(chunks_tokens):
            c = engine.prefill_canonical_chunk(ct)
            canon_chunks.append(c)

        # 2. Benchmark Full Dense Prefill Baseline
        # Reorder chunks: [Chunk 1, Chunk 0, Chunk 3, Chunk 2]
        permuted_order = [(i * 2 + 1) % num_chunks for i in range(num_chunks)]
        permuted_tokens = []
        for idx in permuted_order:
            permuted_tokens.extend(chunks_tokens[idx])
        permuted_ids = torch.tensor([permuted_tokens], device=device, dtype=torch.int64)

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        dense_times = []
        try:
            for _ in range(3):  # 3 iterations
                t0 = time.perf_counter()
                with torch.inference_mode():
                    _ = engine.model(input_ids=permuted_ids)
                torch.cuda.synchronize()
                dense_times.append((time.perf_counter() - t0) * 1000.0)
            t_dense_ms = min(dense_times)
        except torch.OutOfMemoryError:
            print("  -> Full Dense Prefill : OOM (>11 GB VRAM limit)")
            t_dense_ms = float("inf")
            torch.cuda.empty_cache()

        # 3. Benchmark DAG-KV Stitched Prefill + Bridge Repair
        permuted_canon = [canon_chunks[idx] for idx in permuted_order]
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        dag_times = []
        for _ in range(3):
            t0 = time.perf_counter()
            with torch.inference_mode():
                table, doc_tokens = engine.assemble_prompt_dag(permuted_canon, seq_id=f"bench_{total_len}")
                _ = engine.repair_bridge_tokens(table, permuted_canon, top_k_percent=0.08)
            torch.cuda.synchronize()
            dag_times.append((time.perf_counter() - t0) * 1000.0)
        t_dag_ms = min(dag_times)

        speedup = t_dense_ms / t_dag_ms if t_dense_ms != float("inf") else float("inf")
        dense_throughput = total_len / (t_dense_ms / 1000.0)
        dag_throughput = total_len / (t_dag_ms / 1000.0)

        print(f"  -> Full Dense Prefill TTFT : {t_dense_ms:7.2f} ms ({dense_throughput:8.1f} tok/s)")
        print(f"  -> DAG-KV Stitched TTFT   : {t_dag_ms:7.2f} ms ({dag_throughput:8.1f} tok/s)")
        print(f"  -> SPEEDUP                : {speedup:7.2f}x")

        results.append({
            "context_length": total_len,
            "num_chunks": num_chunks,
            "dense_ttft_ms": t_dense_ms,
            "dag_ttft_ms": t_dag_ms,
            "speedup": speedup,
            "dag_throughput_tok_s": dag_throughput,
        })

    print("\n-------------------------------------------------------------")
    print(f"{'Context':<10} | {'Dense TTFT (ms)':<16} | {'DAG-KV (ms)':<12} | {'Speedup':<10}")
    print("-------------------------------------------------------------")
    for r in results:
        print(f"{r['context_length']:<10} | {r['dense_ttft_ms']:<16.2f} | {r['dag_ttft_ms']:<12.2f} | {r['speedup']:<10.2f}x")
    print("-------------------------------------------------------------\n")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", nargs="+", type=int, default=[1024, 2048, 4096])
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    run_ttft_benchmark(lengths=args.lengths, device=args.device)
