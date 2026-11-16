# DAG-KV: Position-Decoupled Graph KV Cache with Query-Side Adjoint RoPE & LSE-Bounded Sparse Bridge Repair

[![Hardware: 2x NVIDIA RTX 2080 Ti (sm_75)](https://img.shields.io/badge/Hardware-2x%20RTX%202080%20Ti%20(sm__75)-76B900?logo=nvidia)](https://developer.nvidia.com/cuda-gpus)
[![Precision: Strict FP16](https://img.shields.io/badge/Precision-Strict%20FP16-blue)](https://pytorch.org)
[![Triton: Turing Optimized (<64KB SRAM)](https://img.shields.io/badge/Triton-SM75%20Optimized-orange)](https://github.com/openai/triton)
[![Tests: 27/27 Passed](https://img.shields.io/badge/Tests-27%2F27%20Passed-brightgreen)](tests/)
[![Paper: PDF](https://img.shields.io/badge/Paper-PDF-red?logo=adobeacrobatreader)](paper/dag_kv_paper.pdf)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

---

## Executive Summary

Existing prefix-caching architectures (e.g., **vLLM Automatic Prefix Caching**, **SGLang RadixAttention**) suffer a catastrophic **0% cache hit rate** when context chunks are dynamically permuted, reordered, or multi-hop composed in Retrieval-Augmented Generation (RAG) and multi-engineer codebase assistant workflows (e.g., `[Sys, Doc_A, Doc_B]` vs `[Sys, Doc_B, Doc_A]`). This failure stems from two fundamental blockers:
1. **RoPE Position Coupling:** Rotary Position Embeddings bake absolute sequence indices directly into cached Key vectors. Mutating cached keys in HBM to shift position offsets destroys Copy-on-Write (CoW) sharing across concurrent requests.
2. **Missing Cross-Chunk Causal Attention:** Chunks prefilled in isolation miss attention interactions with predecessor chunks.

**DAG-KV** solves both problems via three mathematical breakthroughs implemented from scratch for **Turing (`sm_75`)** hardware:
- **Query-Side Adjoint RoPE:** Exploits the orthogonal adjoint identity in $SO(2)$ to counter-rotate query vectors in GPU registers ($q_{\text{adj}} = \mathcal{R}(-\Delta) \tilde{q}$), leaving cached Key blocks **100% read-only and zero-copy in HBM**.
- **Canonical Sink Prefix:** Neutralizes the position-zero attention sink pathology by prepending a 4-token system sink during isolated prefill and slicing it off before registration.
- **Log-Sum-Exp (LSE) Perturbation Bound & Submodular Bridge Repair:** Uses Layer-1 Online Softmax perturbation scalars $\alpha_i = \sigma(\ell_{i \to \text{prior}} - \ell_i^{\text{local}})$ with quality-diversity non-maximum suppression to select sparse Bridge Tokens for ragged micro-prefill, storing repaired keys into request-private **Copy-on-Write (CoW) Shadow Blocks** while reusing **92%–96% of the canonical KV cache untouched**.

> **Research Paper:** Read the complete 3-page manuscript in [`paper/dag_kv_paper.pdf`](paper/dag_kv_paper.pdf).

---

## 3 Core Mathematical Pillars

### Pillar 1: Query-Side Adjoint RoPE in $\mathrm{SO}(2)$

Let document chunk $c$ be prefilled once at local positions $p \in [0, L_c - 1]$ yielding canonical key vectors $\tilde{\mathbf{k}}_p = \mathcal{R}(p)\mathbf{k}_p$. When chunk $c$ is dynamically placed at global offset $\Delta_c$ in a prompt DAG, its tokens sit at global positions $m = p + \Delta_c$.

Because 2D RoPE rotation matrices form a one-parameter subgroup of $SO(2)$:
$$\mathcal{R}(p + \Delta_c) = \mathcal{R}(\Delta_c)\mathcal{R}(p) \quad \text{and} \quad \mathcal{R}(\Delta_c)^T = \mathcal{R}(-\Delta_c)$$

The attention score between query $\mathbf{q}_n$ at global position $n$ and cached key $\tilde{\mathbf{k}}_p$ satisfies the exact adjoint identity:
$$\text{Score}(n, p; \Delta_c) = \big\langle \mathcal{R}(n)\mathbf{q}_n, \, \mathcal{R}(p + \Delta_c)\mathbf{k}_p \big\rangle = \big\langle \mathcal{R}(-\Delta_c)\tilde{\mathbf{q}}_n, \, \tilde{\mathbf{k}}_p \big\rangle$$

```text
Standard System (Mutates Keys in HBM - Destroys Sharing):
  HBM Memory: [ R(0)k0, R(1)k1, ... ]  ---> In-place mutate to [ R(1024)k0, R(1025)k1, ... ] (Corrupts concurrent batch!)

DAG-KV (Query-Side Adjoint - 100% Zero-Copy HBM):
  Query in SRAM:  q_adj = q * cos(Δ) - rotate_half(q) * sin(Δ)
  HBM Memory:     [ R(0)k0, R(1)k1, ... ]  (Never modified, 100% read-only shared pool!)
```

### Pillar 2: Canonical Sink Prefix (Solving Norm Pathology)

Autoregressive language models concentrate massive attention softmax mass ("attention sinks") onto the first 1–4 tokens. If a chunk is prefilled at raw token index 0, its first token absorbs abnormal sink activation norms that corrupt cross-attention representations when shifted to position $\Delta_c > 0$.

**Mechanism:** Every chunk is prefilled with a standardized 4-token system sink prefix (`<|im_start|>system\n`) at indices `[0, 1, 2, 3]`, placing the chunk's payload at canonical positions `[4, ..., L_c + 3]`. After prefill, the 4 sink KV slots are sliced away, storing only the payload KV alongside its local causal Log-Sum-Exp vector $\ell_i^{\text{local}}$.

### Pillar 3: Layer-1 LSE Perturbation Bound & Submodular Bridge Selection

When concatenating reordered chunks `[Chunk_A, Chunk_B]`, tokens in `Chunk_B` missed attending to `Chunk_A`. Via Online Softmax decomposition:
$$\alpha_i = \frac{\exp(\ell_{i \to \text{prior}})}{\exp(\ell_i^{\text{local}}) + \exp(\ell_{i \to \text{prior}})} = \sigma\big(\ell_{i \to \text{prior}} - \ell_i^{\text{local}}\big)$$

By the Cauchy-Schwarz bound on attention outputs:
$$\|\mathbf{o}_i^{\text{exact}} - \mathbf{o}_i^{\text{cached}}\|_2 \le 2\alpha_i \max_j \|\mathbf{v}_j\|_2$$

Rather than greedy top-$k$ selection—which redundantly clusters on adjacent tokens of the same line—DAG-KV maximizes a spatial quality-diversity objective via 1D neighborhood non-maximum suppression ($r=2$), selecting **4.0% bridge tokens on Stage 0** (`cuda:0`, Layers 0–13) and pruning to **1.5% before crossing PCIe to Stage 1** (`cuda:1`, Layers 14–27). Repaired KVs are written into request-private **Copy-on-Write (CoW) Shadow Blocks**, achieving **4.66x faster stitching** while reusing **96% of the canonical KV cache untouched**.

---

## Turing (`sm_75`) Hardware Implementation Contract

Designed and verified on dual **NVIDIA GeForce RTX 2080 Ti** (Turing `sm_75`, 11 GB GDDR6 per GPU):

| Constraint | Enforcement | File / Mechanism |
| :--- | :--- | :--- |
| **STRICT BAN ON `bfloat16`** | Turing SMs lack native BF16 arithmetic. All weights, KV cache, and activations use `torch.float16` (`fp16`). | `configs/hw_rtx2080ti.yaml`, `dag_kv/models/qwen2_dag_patch.py` |
| **FP32 Accumulators in Triton** | All vector sums and dot products accumulate in `tl.float32`. | `dag_kv/kernels/triton_adjoint_paged_attn.py`, `dag_kv/kernels/triton_sparse_stitch.py` |
| **SRAM Shared Memory Limit** | SM75 hardware cap is 64 KB (65,536 bytes) per SM. Triton tile sizes: `BLOCK_M=64`, `BLOCK_N=16`, `num_stages=2`. | Verified: `test_turing_sm75_shared_memory_limits` measures **2,048 bytes** (<3.2% of 64KB limit). |
| **No Ampere Flash-Attn** | Avoids Ampere-only wheels (`flash_attn` v2+ crashes on SM75). Uses custom Triton kernels and PyTorch SDPA. | `tests/test_triton_sm75.py` |
| **Dual-GPU Pipeline & Overlap** | `cuda:1` runs asynchronous prefill or Stage 1 pipeline; PCIe DMA stream transfers blocks with zero host stalls. | `dag_kv/worker/dual_gpu_pipeline.py`, `benchmarks/bench_pipeline_overlap.py` |

---

## Repository Architecture

```text
dag-kv/
├── configs/
│   └── hw_rtx2080ti.yaml             # Turing SM75 tile configs, memory constraints, dual-GPU pipeline
├── dag_kv/
│   ├── core/
│   │   ├── rope_adjoint.py           # Adjoint RoPE R(-Delta) implementation and mathematical proofs
│   │   ├── sink_canonicalizer.py     # 4-token sink prefix canonicalization & SHA256 hashing
│   │   ├── lse_probe.py              # Layer-1 LSE perturbation bound alpha_i & top-k% selector
│   │   └── block_manager.py          # Paged Block Pool, OffsetBlockTable & CoW Shadow Blocks
│   ├── kernels/
│   │   ├── triton_adjoint_paged_attn.py # Turing SM75-tuned PagedAttention decode with R(-Delta_b)q
│   │   └── triton_sparse_stitch.py      # Fused ragged micro-prefill kernel for Bridge Tokens
│   ├── models/
│   │   └── qwen2_dag_patch.py        # Qwen2DAGKVEngine wrapping Qwen2.5-3B-Instruct in FP16
│   ├── worker/
│   │   └── dual_gpu_pipeline.py      # Dual-GPU pipeline: CUDA:1 Prefill -> DMA -> CUDA:0 Engine
│   └── server.py                     # OpenAI-compatible /v1/chat/completions HTTP server
├── paper/
│   ├── dag_kv_paper.pdf              # Compiled 3-page research paper PDF (ready for publication)
│   └── main.tex                      # LaTeX manuscript source
├── assets/
│   └── dag_kv_benchmarks.png         # High-resolution benchmark overview chart
├── tests/
│   ├── test_core_components.py       # Unit tests for Sink Canonicalizer, LSE Probe, BlockManager
│   ├── test_rope_adjoint.py          # 13 tests: sweeping dims 64/128, bases 1e4/1e6, FP16/FP32
│   ├── test_triton_sm75.py           # 9 tests: Triton kernels vs PyTorch gold standard & 64KB SRAM check
│   ├── test_dual_gpu_pipeline.py     # Disaggregated dual-GPU prefill & stitch test
│   └── test_sparse_bridge_e2e.py     # End-to-end Qwen2.5-3B stitched vs dense prefill verification
└── benchmarks/
    ├── bench_concurrent_vram.py      # Concurrency & batch capacity benchmark (11 GB VRAM)
    ├── bench_ttft_speedup.py         # Time-To-First-Token speedup across 512..4096 context lengths
    ├── eval_multihop_rag.py          # Multi-hop RAG evaluation under chunk reordering
    ├── bench_dual_gpu_pipeline.py    # Dual-GPU NUMA disaggregated pipeline & p99 tail latency
    ├── bench_7b_realworld_rag.py     # Real-world 7B scale benchmark sharded across 2x RTX 2080 Ti
    ├── bench_multiuser_codebase.py   # Multi-engineer codebase assistant benchmark (12 sessions)
    ├── bench_int8_triton_kv.py       # Fused INT8 Triton KV cache benchmark
    ├── bench_batched_lru_serving.py  # Multi-tenant B=4 batched decode with LRU eviction
    ├── bench_ast_delta_cache.py      # AST function-level Git-diff live edit benchmark
    ├── bench_pipeline_overlap.py     # Dual-GPU ping-pong micro-batch pipeline overlap
    ├── bench_submodular_bridge.py    # Submodular quality-diversity bridge repair benchmark
    ├── bench_openai_server.py        # End-to-end /v1/chat/completions HTTP server test
    └── generate_plots.py             # Generates benchmark visualization plots
```

---

## Empirical Benchmark Results

![DAG-KV Architecture and Empirical Benchmarks](assets/dag_kv_benchmarks.png)

### Summary of Performance Gains on $2\times$ NVIDIA RTX 2080 Ti (`Qwen2.5-7B-Instruct`)

| Dimension | Standard Serving Baseline (vLLM / SGLang) | DAG-KV Stitched Engine (Ours) | Measured Gain |
| :--- | :--- | :--- | :---: |
| **Context Stitch Latency (4K Toks)** | 1,441.8 ms (Dense 7B Prefill) | **309.5 ms** (Submodular $4\% \to 1.5\%$) | **4.66x Faster** |
| **Multi-User KV VRAM (12 Sessions)** | 2,522.1 MB (Private copies) | **258.2 MB** (Shared Pool + INT8) | **9.8x Less VRAM** |
| **Decode Throughput ($B=4\times 2$)** | 23.1 tok/s (Sequential $B=1$) | **100.4 tok/s** (Ping-Pong Overlap) | **4.35x Throughput** |
| **Live IDE Save Re-Indexing** | 994.4 ms (Full re-prefill) | **79.3 ms** (AST Function-Delta) | **12.5x Faster** |
| **Reordered File Cache Hit Rate** | **0.0%** (Prefix cache miss) | **100.0%** (Zero-copy canonical hit) | **$\infty$** |
| **Codebase QA Accuracy** | N/A | **100.0% (12/12 Exact Answers)** | **1.000 F1** |

---

### Detailed Benchmark Reports

#### 1. Concurrency & VRAM Capacity (`bench_concurrent_vram.py`)
*Hardware: NVIDIA GeForce RTX 2080 Ti (11 GB VRAM), Model: `Qwen/Qwen2.5-3B-Instruct` (FP16, 5.75 GB base weights).*

| Architecture | Canonical Sharing | Per-Request VRAM | Max Concurrent Streams ($B_{\max}$) | Concurrency Boost | VRAM Reduction |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Mutated Private KV Baseline** | None (Private Copies) | 135.00 MB | 38 streams | 1.0x (Baseline) | 0.0% |
| **DAG-KV Zero-Copy Pool** | **Yes (Shared + 8% CoW)** | **10.80 MB** | **474 streams** | **12.5x** | **88.9%** |

#### 2. Time-To-First-Token (TTFT) Speedup (`bench_ttft_speedup.py`)
*Measured with single-pass fused Triton Ragged Micro-Prefill (`_triton_sparse_bridge_micro_prefill_kernel`) on `Qwen/Qwen2.5-3B-Instruct` (FP16).*

| Context Length | Chunks | Dense Prefill TTFT | DAG-KV Stitched TTFT | Effective Throughput | TTFT Speedup |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **512 tokens** | 4 | 98.07 ms | **31.48 ms** | **16,266.4 tok/s** | **3.12x** |
| **1024 tokens** | 4 | 230.19 ms | **56.67 ms** | **18,070.4 tok/s** | **4.06x** |
| **2048 tokens** | 4 | 600.14 ms | **165.97 ms** | **12,339.5 tok/s** | **3.62x** |
| **4096 tokens** | 4 | **OOM (>11 GB)** | **591.58 ms** | **6,923.9 tok/s** | **$\infty$ (Solves OOM)** |

#### 3. Multi-Hop RAG Accuracy & Pillar Ablation (`eval_multihop_rag.py`)

| Evaluation Metric | Standard Prefix Cache (vLLM / SGLang) | DAG-KV Stitched Engine (Ours) |
| :--- | :---: | :---: |
| **Cache Hit Rate on Reordered Chunks (`[B, A]`)** | **0.0%** (Miss from index 0) | **100.0%** (Zero-Copy Chunk Hit) |
| **Canonical KV Cache Token Reuse** | 0.0% | **92.0%** (8.0% LSE Bridge Repaired) |
| **Logit Cosine Similarity vs. Dense Oracle** | N/A (Recomputes 100%) | **0.9937 (99.37% Similarity)** |
| **KL Divergence vs. Dense Oracle ($\mathcal{D}_{\text{KL}}$)** | 0.0201 nats (Unshifted Reuse) | **0.0011 nats (19.1x Lower Drift)** |
| **Top-1 Next-Token Agreement vs. Dense Oracle** | N/A | **100.0% (6/6 Permutations)** |
| **Multi-Hop QA Token F1 Score (`[A, B]` & `[B, A]`)** | N/A | **1.000 (Exact Match)** |

#### 4. Dual-GPU Disaggregated Pipeline & p99 Tail Latency (`bench_dual_gpu_pipeline.py`)
*Topology: Dual-Socket NUMA. `cuda:1` runs asynchronous canonical chunk prefill and transfers KV + LSE tensors via Page-Locked (`pin_memory=True`) Host DMA to `cuda:0`.*

| Pipeline Mode | Total Workload Time | p50 Median TTFT | p99 Tail TTFT | Tail Latency Reduction |
| :--- | :---: | :---: | :---: | :---: |
| **1. Single-GPU Interleaved (`cuda:0` only)** | 1839.48 ms | 78.64 ms | 303.91 ms (HoL Stall) | 1.00x (Baseline) |
| **2. Dual-GPU Disaggregated (`cuda:1` $\to$ `cuda:0`)** | **994.45 ms (1.85x Faster)** | **83.70 ms** | **86.26 ms** | **3.52x Lower p99 Spike** |

#### 5. Real-World 7B Scale on 2x RTX 2080 Ti (`bench_7b_realworld_rag.py`)
*Model: `Qwen2.5-7B-Instruct` (~15.2 GB in FP16), pipeline-sharded across dual GPUs (Shard 0 on `cuda:0` for Layers 0..13; Shard 1 on `cuda:1` for Layers 14..27).*

| Evaluation Metric | Qwen2.5-7B-Instruct (Pipeline-Sharded 2x RTX 2080 Ti) |
| :--- | :---: |
| **VRAM Allocation** | GPU 0: **6.80 GB** \| GPU 1: **8.73 GB** (Fits in 11GB HBM) |
| **Multi-Hop RAG Dataset** | 5-Chunk Contexts (2 Evidence + 3 Distractors, 4 Permutations = 20 Runs) |
| **Token F1 Score across 20 Permuted Runs** | **1.000 (Exact Retrieval)** |
| **Top-1 Next-Token Agreement vs. 7B Oracle** | **100.0%** |
| **Mean KL Divergence (DAG-KV 7B)** | **0.0248 nats** (Near-zero distributional shift) |

#### 6. Multi-Engineer Codebase Assistant (`bench_multiuser_codebase.py`)
*Simulates 3 concurrent software engineers querying live repository files across 12 permuted file orderings on `Qwen2.5-7B-Instruct`.*

| Production Metric | Standard Prefix Cache (vLLM / SGLang) | DAG-KV Shared Codebase Pool (7B) |
| :--- | :---: | :---: |
| **Shared Codebase File Cache Hit Rate** | **1.8% – 3.0%** (Fails on reordered files) | **100.0%** (92% Zero-Copy + 8% Bridge) |
| **Mean 4-File Stitch Latency (2.8K / 4.0K Toks)** | 1217.3 ms / 1443.3 ms (OOMs without chunking) | **455.5 ms (`2.67x`) / 755.4 ms (`1.91x`)** |
| **Codebase QA Accuracy (Token F1 Score)** | N/A | **1.000 (12/12 Exact Code Answers)** |
| **Top-1 Next-Token Agreement vs. 7B Oracle** | N/A | **100.0% (12/12 Queries, CosSim = 0.9318)** |
| **Multi-User Team KV Cache VRAM (12 Sessions)** | 2522.1 MB (Duplicated per user) | **515.9 MB (4.9x Less VRAM)** |

#### 7. Algorithmic & Systems Upgrades
*Verified on `Qwen2.5-7B-Instruct` across 2x NVIDIA RTX 2080 Ti GPUs (`sm_75`).*

| System Upgrade | Benchmark Script | Verified Result |
| :--- | :--- | :--- |
| **Submodular Diversity Bridge (`4%` $\to$ `1.5%`)** | `benchmarks/bench_submodular_bridge.py` | **309.5 ms** stitch (**4.66x speedup** vs. Dense 7B), **100% QA Accuracy (6/6)** |
| **Dual-GPU Ping-Pong Overlap (`cuda:0` $\parallel$ `cuda:1`)** | `benchmarks/bench_pipeline_overlap.py` | **100.4 tok/s** across 8 concurrent streams (**4.35x throughput**), **100% Bitwise Match** |
| **AST Function-Level Git-Diff Delta Cache** | `benchmarks/bench_ast_delta_cache.py` | **167.6 ms** mean edit re-index (**5.93x mean**, up to **12.50x / 79.3 ms**), **100% Diff QA** |
| **Fused INT8 Triton KV Cache** | `benchmarks/bench_int8_triton_kv.py` | **9.8x Lower Multi-User VRAM** (**258.2 MB** vs. 2,522.1 MB), **0.9984** Cosine Similarity |
| **Batched Decode ($B=4$) & LRU Eviction** | `benchmarks/bench_batched_lru_serving.py` | **73.5 tok/s** (**3.18x throughput**), **0 OOMs** across 19 files with auto-eviction |
| **Live OpenAI-Compatible HTTP Server** | `dag_kv/server.py` | **100.0% File Cache Hit Rate**, **2.68x End-to-End HTTP Speedup** |

---

## Production OpenAI-Compatible HTTP Server

DAG-KV includes an OpenAI-compatible `/v1/chat/completions` server (`dag_kv/server.py`) designed to drop directly into developer tools (e.g., Cursor, Continue.dev):

```bash
# Start the DAG-KV server on port 8000
python3 dag_kv/server.py --port 8000 --model Qwen/Qwen2.5-7B-Instruct
```

Query the endpoint with standard curl or OpenAI client libraries:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct-DAG-KV",
    "messages": [
      {"role": "user", "content": "### FILE: dag_kv/core/block_manager.py\n...\n### FILE: dag_kv/core/rope_adjoint.py\n...\nQuestion: What is the default block size?"}
    ],
    "max_tokens": 16
  }'
```

---

## Quickstart & Reproducibility Guide

### Environment Setup
```bash
git clone https://github.com/kaushalnandaniya/dag-kv.git
cd dag-kv

# Install in editable mode
pip install -e .
```

### Running the Test Suite (27/27 Tests)
```bash
PYTHONPATH=. pytest tests/ -v
```
Expected output:
```text
tests/test_core_components.py::test_sink_canonicalizer PASSED            [  3%]
tests/test_core_components.py::test_lse_probe_and_bridge_selection PASSED [  7%]
tests/test_core_components.py::test_block_manager_cow_and_assembly PASSED [ 11%]
tests/test_dual_gpu_pipeline.py::test_dual_gpu_disaggregated_prefill_and_stitch PASSED [ 14%]
tests/test_rope_adjoint.py::test_rope_adjoint_identity_scalar ... PASSED [ 44%]
tests/test_rope_adjoint.py::test_rope_adjoint_batched_multihead ... PASSED [ 59%]
tests/test_rope_adjoint.py::test_rotate_half_orthogonality PASSED        [ 62%]
tests/test_sparse_bridge_e2e.py::test_qwen2_dag_kv_e2e_stitching_and_repair PASSED [ 66%]
tests/test_triton_sm75.py::test_triton_adjoint_paged_attention_accuracy ... PASSED [ 88%]
tests/test_triton_sm75.py::test_triton_sparse_bridge_attention_accuracy ... PASSED [ 96%]
tests/test_triton_sm75.py::test_turing_sm75_shared_memory_limits PASSED  [100%]

============================= 27 passed in 28.70s ==============================
```

### Running the Production Benchmarks
```bash
# 1. Concurrency & VRAM Capacity Benchmark
PYTHONPATH=. python3 benchmarks/bench_concurrent_vram.py

# 2. Time-To-First-Token (TTFT) Speedup Benchmark
PYTHONPATH=. python3 benchmarks/bench_ttft_speedup.py --lengths 512 1024 2048 4096

# 3. Multi-Hop RAG Evaluation across Permutations
PYTHONPATH=. python3 benchmarks/eval_multihop_rag.py

# 4. Multi-Engineer Codebase Assistant (7B Model)
PYTHONPATH=. python3 benchmarks/bench_multiuser_codebase.py

# 5. Submodular Diversity Bridge Repair Benchmark
PYTHONPATH=. python3 benchmarks/bench_submodular_bridge.py

# 6. Dual-GPU Ping-Pong Pipeline Overlap Benchmark
PYTHONPATH=. python3 benchmarks/bench_pipeline_overlap.py

# 7. AST Function-Delta Live Edit Benchmark
PYTHONPATH=. python3 benchmarks/bench_ast_delta_cache.py

# 8. OpenAI-Compatible Server Benchmark
PYTHONPATH=. python3 benchmarks/bench_openai_server.py
```

### Python API Usage
```python
from dag_kv.models.qwen2_dag_patch import Qwen2DAGKVEngine

# 1. Initialize engine on RTX 2080 Ti (sm_75, FP16)
engine = Qwen2DAGKVEngine.from_pretrained("Qwen/Qwen2.5-3B-Instruct", device="cuda:0")

# 2. Prefill documents into canonical position-zero storage
doc1 = "The Eiffel Tower was designed by Gustave Eiffel for the 1889 World Fair in Paris.\n"
doc2 = "The 1889 World Fair was held to celebrate the centennial of the French Revolution.\n"
chunk1 = engine.prefill_canonical_chunk(doc1)
chunk2 = engine.prefill_canonical_chunk(doc2)

# 3. Dynamically compose permuted DAG context [Doc 2, Doc 1] with zero-copy reuse
gen_tokens, text = engine.generate(
    chunks=[chunk2, chunk1],
    prompt_suffix="\nQuestion: What historical event was the 1889 World Fair celebrating?\nAnswer: The event was the ",
    max_new_tokens=20,
)
print("Answer:", text)
```

---

## Citation & License

```bibtex
@software{dag_kv_2026,
  title={{DAG-KV: Position-Decoupled Graph KV Cache with Query-Side Adjoint RoPE & LSE-Bounded Sparse Bridge Repair}},
  author={Nandaniya, Kaushal and Collaborators},
  year={2026},
  url={https://github.com/kaushalnandaniya/dag-kv}
}
```

Licensed under the [Apache 2.0 License](LICENSE).
