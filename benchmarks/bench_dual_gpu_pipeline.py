"""Benchmark: Dual-GPU Disaggregated Pipeline (cuda:1 Async Prefill -> cuda:0 Stitch/Decode).

Simulates a live streaming RAG server handling interleaved document ingestion and
multi-hop user queries across 4 rounds (each round ingests 2 new 512-token documents
while serving user queries over previously ingested chunks):

- Mode 1 (Single-GPU Interleaved on cuda:0):
  GPU 0 must pause query serving to prefill new documents on the same GPU (Head-of-Line Blocking).
- Mode 2 (Dual-GPU Disaggregated on cuda:1 + cuda:0):
  GPU 1 asynchronously prefills incoming documents in the background and streams them via
  Pinned Host DMA to GPU 0 while GPU 0 simultaneously stitches and decodes active queries.
"""

import time
import torch
from dag_kv.worker.dual_gpu_pipeline import DualGPUPipeline


def run_dual_gpu_benchmark(
    num_rounds: int = 4,
    chunk_tokens_len: int = 512,
    queries_per_round: int = 3,
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
):
    print("\n=====================================================================")
    print("  DAG-KV DUAL-GPU DISAGGREGATED PIPELINE BENCHMARK (2x RTX 2080 Ti)")
    print(f"  Topology: GPU 0 (Inference Engine) + GPU 1 (Async Prefill Worker)")
    print(f"  Workload: {num_rounds} Rounds x (2 New {chunk_tokens_len}-tok Docs + {queries_per_round} Stitched Queries)")
    print("=====================================================================\n")

    assert torch.cuda.device_count() >= 2, "Requires 2 GPUs (cuda:0 and cuda:1)"

    pipe = DualGPUPipeline(
        model_name_or_path=model_name,
        inference_device="cuda:0",
        prefill_device="cuda:1",
    )
    tokenizer = pipe.engine.tokenizer

    base_corpus = (
        "Distributed retrieval-augmented generation systems ingest continuous streams of enterprise "
        "documents, source code files, and telemetry logs. Decoupling canonical chunk prefill onto "
        "a dedicated secondary GPU prevents compute-bound prefill kernels from stalling latency-sensitive "
        "DAG stitching and autoregressive decoding on the primary serving GPU. "
    ) * 40
    raw_ids = tokenizer.encode(base_corpus, add_special_tokens=False)

    # Build unique document chunks for the workload
    all_doc_tokens = []
    for i in range((num_rounds + 1) * 2):
        offset = i * 37
        doc_slice = (raw_ids[offset:] + raw_ids[:offset])[:chunk_tokens_len]
        # Make each chunk content-unique via a distinct header token sequence
        header = tokenizer.encode(f"Document Section {i + 100}: ", add_special_tokens=False)
        all_doc_tokens.append(header + doc_slice[: chunk_tokens_len - len(header)])

    query_suffix = "\nQuestion: Summarize the core system mechanism.\nAnswer:"

    # Seed initial 2 chunks
    initial_c0 = pipe.prefill_chunk(all_doc_tokens[0])
    initial_c1 = pipe.prefill_chunk(all_doc_tokens[1])

    # Warmup
    _ = pipe.stitch_and_generate([initial_c0, initial_c1], prompt_suffix=query_suffix, max_new_tokens=4)
    torch.cuda.synchronize("cuda:0")
    torch.cuda.synchronize("cuda:1")

    # -------------------------------------------------------------------------
    # 1. Benchmark Single-GPU Interleaved (Everything on cuda:0)
    # -------------------------------------------------------------------------
    pipe.engine._chunk_cache.clear()
    pipe.engine.block_manager.canonical_registry.clear()
    c_prev_0 = pipe.engine.prefill_canonical_chunk(all_doc_tokens[0])
    c_prev_1 = pipe.engine.prefill_canonical_chunk(all_doc_tokens[1])

    torch.cuda.synchronize("cuda:0")
    t0_single = time.perf_counter()
    single_query_latencies = []

    for r in range(1, num_rounds + 1):
        new_doc_a = all_doc_tokens[r * 2]
        new_doc_b = all_doc_tokens[r * 2 + 1]

        # On a single GPU, incoming documents must be prefilled on cuda:0 before/between queries
        tq0 = time.perf_counter()
        next_c0 = pipe.engine.prefill_canonical_chunk(new_doc_a)
        next_c1 = pipe.engine.prefill_canonical_chunk(new_doc_b)

        for q_idx in range(queries_per_round):
            t_q_start = time.perf_counter() if q_idx > 0 else tq0
            order = [c_prev_0, c_prev_1] if q_idx % 2 == 0 else [c_prev_1, c_prev_0]
            table, doc_toks = pipe.engine.assemble_prompt_dag(order, seq_id=f"s_{r}_{q_idx}")
            pipe.engine.repair_bridge_tokens(table, order, top_k_percent=0.08)
            _ = pipe.engine.append_and_decode(table, token_id=doc_toks[-1], current_pos=len(doc_toks))
            torch.cuda.synchronize("cuda:0")
            single_query_latencies.append((time.perf_counter() - t_q_start) * 1000.0)

        c_prev_0, c_prev_1 = next_c0, next_c1

    total_single_ms = (time.perf_counter() - t0_single) * 1000.0

    # -------------------------------------------------------------------------
    # 2. Benchmark Dual-GPU Disaggregated (cuda:1 Async Prefill || cuda:0 Stitch)
    # -------------------------------------------------------------------------
    pipe.prefiller._chunk_cache.clear()
    c_prev_0 = pipe.prefill_chunk(all_doc_tokens[0])
    c_prev_1 = pipe.prefill_chunk(all_doc_tokens[1])

    torch.cuda.synchronize("cuda:0")
    torch.cuda.synchronize("cuda:1")
    t0_dual = time.perf_counter()
    dual_query_latencies = []

    for r in range(1, num_rounds + 1):
        new_doc_a = all_doc_tokens[r * 2]
        new_doc_b = all_doc_tokens[r * 2 + 1]

        # Launch non-blocking background prefill on GPU 1 (cuda:1)
        fut_a = pipe.prefill_chunk_async(new_doc_a)
        fut_b = pipe.prefill_chunk_async(new_doc_b)

        # Meanwhile, GPU 0 immediately serves all user queries with ZERO prefill stall!
        for q_idx in range(queries_per_round):
            t_q_start = time.perf_counter()
            order = [c_prev_0, c_prev_1] if q_idx % 2 == 0 else [c_prev_1, c_prev_0]
            table, doc_toks = pipe.engine.assemble_prompt_dag(order, seq_id=f"d_{r}_{q_idx}")
            pipe.engine.repair_bridge_tokens(table, order, top_k_percent=0.08)
            _ = pipe.engine.append_and_decode(table, token_id=doc_toks[-1], current_pos=len(doc_toks))
            torch.cuda.synchronize("cuda:0")
            dual_query_latencies.append((time.perf_counter() - t_q_start) * 1000.0)

        # Collect prefilled chunks from GPU 1 for the next round
        c_prev_0 = pipe.register_transferred_chunk(fut_a.result())
        c_prev_1 = pipe.register_transferred_chunk(fut_b.result())

    total_dual_ms = (time.perf_counter() - t0_dual) * 1000.0

    single_p50 = sorted(single_query_latencies)[len(single_query_latencies) // 2]
    single_p99 = max(single_query_latencies)
    dual_p50 = sorted(dual_query_latencies)[len(dual_query_latencies) // 2]
    dual_p99 = max(dual_query_latencies)

    wall_speedup = total_single_ms / total_dual_ms
    p99_reduction = single_p99 / dual_p99

    print("\n----------------------------------------------------------------------------------")
    print(f"{'Pipeline Mode':<34} | {'Total Time':<12} | {'p50 TTFT':<10} | {'p99 Tail TTFT':<14}")
    print("----------------------------------------------------------------------------------")
    print(f"{'1. Single-GPU Interleaved (cuda:0)':<34} | {total_single_ms:9.2f} ms | {single_p50:7.2f} ms | {single_p99:11.2f} ms")
    print(f"{'2. Dual-GPU Disaggregated (0 + 1)':<34} | {total_dual_ms:9.2f} ms | {dual_p50:7.2f} ms | {dual_p99:11.2f} ms")
    print("----------------------------------------------------------------------------------")
    print(f"End-to-End Pipeline Speedup        : {wall_speedup:.2f}x")
    print(f"p99 Tail TTFT Latency Reduction    : {p99_reduction:.2f}x Lower Tail Spike (Zero HoL Blocking!)\n")


if __name__ == "__main__":
    run_dual_gpu_benchmark()
