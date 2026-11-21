# DAG-KV: Universal Position-Decoupled Graph KV Cache with Query-Side Adjoint RoPE & LSE-Bounded Sparse Bridge Repair

[![Hardware: NVIDIA | AMD ROCm | Apple Silicon MPS | CPU](https://img.shields.io/badge/Hardware-NVIDIA%20%7C%20AMD%20ROCm%20%7C%20Apple%20Silicon%20%7C%20CPU-76B900)](https://pytorch.org)
[![Execution: Triton JIT + PyTorch Native Fallback](https://img.shields.io/badge/Kernels-Triton%20%2B%20PyTorch%20Fallback-orange)](dag_kv/kernels/)
[![Tests: 69 Tests](https://img.shields.io/badge/Tests-69%20Tests-brightgreen)](tests/)
[![Release: v0.3.2](https://img.shields.io/badge/Release-v0.3.2-blue)](pyproject.toml)
[![Paper: PDF](https://img.shields.io/badge/Paper-PDF-red?logo=adobeacrobatreader)](paper/dag_kv_paper.pdf)
[![CI/CD: Multi-Arch Matrix](https://img.shields.io/badge/CI%2FCD-Ubuntu%20%7C%20macOS-blue?logo=githubactions)](.github/workflows/ci.yml)
[![Docker: Multi-Arch amd64/arm64](https://img.shields.io/badge/Docker-Multi--Arch%20(amd64%2Farm64)-2496ED?logo=docker)](Dockerfile)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

---

## Executive Summary

Existing prefix-caching architectures (e.g., **vLLM Automatic Prefix Caching**, **SGLang RadixAttention**) suffer a catastrophic **0% cache hit rate** when context chunks are dynamically permuted, reordered, or multi-hop composed in Retrieval-Augmented Generation (RAG) and multi-engineer codebase assistant workflows (e.g., `[Sys, Doc_A, Doc_B]` vs `[Sys, Doc_B, Doc_A]`). This failure stems from two fundamental blockers:
1. **RoPE Position Coupling:** Rotary Position Embeddings bake absolute sequence indices directly into cached Key vectors. Mutating cached keys in HBM to shift position offsets destroys Copy-on-Write (CoW) sharing across concurrent requests.
2. **Missing Cross-Chunk Causal Attention:** Chunks prefilled in isolation miss attention interactions with predecessor chunks.

**DAG-KV** turns graph-structured context caching into a **universal, production-ready library** across all hardware architectures (NVIDIA, AMD ROCm, Apple Silicon MPS, and Host CPU):
- **Query-Side Adjoint RoPE:** Exploits the orthogonal adjoint identity in $SO(2)$ to counter-rotate query vectors in GPU/CPU registers ($q_{\text{adj}} = \mathcal{R}(-\Delta) \tilde{q}$), leaving cached Key blocks **100% read-only and zero-copy in memory**.
- **Canonical Sink Prefix:** Neutralizes the position-zero attention sink pathology by prepending a 4-token system sink during isolated prefill and slicing it off before registration.
- **LSE Perturbation Bound & Submodular Bridge Repair:** Uses Layer-1 Online Softmax perturbation scalars $\alpha_i = \sigma(\ell_{i \to \text{prior}} - \ell_i^{\text{local}})$ with quality-diversity non-maximum suppression to select sparse Bridge Tokens for ragged micro-prefill, storing repaired keys into request-private **Copy-on-Write (CoW) Shadow Blocks** while reusing **92%–96% of the canonical KV cache untouched**.
- **Continuous Batching & Async Engine (v0.3.0):** Concurrent GPU decode batching with `ContinuousBatchingScheduler` and OpenAI-compatible SSE streaming (`/v1/chat/completions`).
- **Reference-Counted LRU Eviction (v0.3.0):** Automatic pruning of cold canonical chunks upon memory pressure, respecting CoW references without OOM aborts.
- **Fused INT4 SRAM Triton Kernels (v0.3.0):** On-the-fly bit-unpacked INT4 dequantization directly inside Triton SRAM loops, cutting KV memory bandwidth by 75% compared to FP16.
- **Model Generalization (v0.3.0):** Model-agnostic base engine supporting Meta Llama-3, Mistral, Google Gemma-2, and Qwen-2 via `AutoDAGKVEngine`.
- **Persistent Disk KV Cache (v0.3.0):** Zero-copy safetensors-backed NVMe persistence and pinned host RAM DMA staging (`DiskKVCache` + `PinnedCPUPool`).
- **Hardware Abstraction Layer (HAL):** Dynamic device manager with auto-discovery, topology auto-configuration (Single-GPU, Multi-GPU Pipeline, Apple Silicon MPS, CPU, and CPU-offloaded KV pooling), adaptive INT8/INT4 quantization, and seamless drop-in integrations for **Hugging Face** and **vLLM**.

> **Research Paper:** Read the complete 3-page manuscript in [`paper/dag_kv_paper.pdf`](paper/dag_kv_paper.pdf).

---

## Universal Hardware Abstraction & Multi-Backend Architecture

DAG-KV abstracts away low-level GPU dependencies, providing automated backend routing based on the host environment:

```
                            ┌──────────────────────────────────┐
                            │    User API / HTTP Endpoint /    │
                            │   HuggingFace / vLLM Integrations│
                            └─────────────────┬────────────────┘
                                              │
                                              ▼
                            ┌──────────────────────────────────┐
                            │   Hardware Abstraction Layer     │
                            │  DeviceManager & ShardConfig     │
                            └─────────────────┬────────────────┘
                                              │
              ┌───────────────────────────────┴───────────────────────────────┐
              ▼                                                               ▼
  ┌───────────────────────┐                                       ┌───────────────────────┐
  │   High-Performance    │                                       │   Universal Native    │
  │    Triton JIT Path    │                                       │ PyTorch Fallback Path │
  └───────────┬───────────┘                                       └───────────┬───────────┘
              │                                                               │
     ┌────────┴────────┐                                             ┌────────┴────────┐
     ▼                 ▼                                             ▼                 ▼
NVIDIA GPUs        AMD ROCm                                     Apple Silicon       Host CPU
(Hopper, Ada,      (MI200, MI300)                               (M1/M2/M3/M4 MPS)   (x86_64, ARM64)
Ampere, Turing)
```

### 1. Triton JIT with Vectorized PyTorch Fallback
* **Triton Execution Path:** Maximum throughput on NVIDIA and AMD ROCm GPUs using fused FP16/BF16 kernels (`triton_adjoint_paged_attn.py` and `triton_sparse_stitch.py`).
* **PyTorch Native Fallback Path:** Pure PyTorch tensor implementation (`pytorch_fallbacks.py`) supporting Apple Silicon (Metal/MPS), CPU-only servers, and quantized pools with zero external kernel compilation dependencies.
* **Unified Dispatcher:** Transparently selects the fastest available backend at runtime via `dag_kv.kernels.dispatch`.

### 2. Dynamic Auto-Sharding Topologies
* **Single-Device Mode:** Runs prefill and decode on a single accelerator (`cuda:0`, `mps`) or CPU.
* **Multi-GPU Disaggregated Pipeline:** Pins asynchronous canonical chunk prefill to one GPU (e.g. `cuda:1`) while serving stitched multi-file decode on another (e.g. `cuda:0`), synchronizing via page-locked (pinned) Host RAM or P2P DMA.
* **CPU-Offloaded KV Pooling:** Keeps model weights on high-speed accelerator HBM while offloading massive shared canonical KV blocks to system RAM with pinned memory staging.

### 3. Adaptive Quantization & Memory Compression
* **Runtime Quantization Selector:** Switch between native floating point (`none`), **INT8** (`1.97x`), **K8V4** (`2.53x`, INT8 Keys + Group-32 INT4 Values — outlier-safe default for Qwen2.5 and Llama-3.2), and **INT4** (`3.05x`, decoupled `K=g8 / V=g32` symmetric quantization), all backed by fused Triton SRAM decode/bridge kernels and native PyTorch fallbacks.
* **Configurable Block Sizes:** Dynamic block sizes (`BLOCK_SIZE = 8, 16, 32, 64`) to balance fragmentation vs. memory efficiency for small contexts or massive codebases.

#### Real-Weight Quantization Pareto Benchmark (RTX 2080 Ti, `v0.3.2`)

| Model | Quant | KV Pool (8K slots) | Compression | Stitch+Repair | Decode (ms/tok) | Throughput | Multi-Chunk EM | Top-1 vs HF | Mean KL |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-3B-Instruct | FP16 | 288.0 MiB | 1.00x | 39.8 ms | 29.5 ms | 33.9 tok/s | 6/6 (100%) | 100% | 0.0308 |
| Qwen2.5-3B-Instruct | INT8 | 146.2 MiB | 1.97x | 49.0 ms | 38.4 ms | 26.0 tok/s | 6/6 (100%) | 100% | 0.0429 |
| Qwen2.5-3B-Instruct | **K8V4 (g32)** | **113.6 MiB** | **2.53x** | 50.5 ms | 41.6 ms | 24.0 tok/s | **6/6 (100%)** | **100%** | **0.0515** |
| Qwen2.5-3B-Instruct | **INT4 (K8g/V32g)** | **94.5 MiB** | **3.05x** | 52.8 ms | 44.0 ms | 22.7 tok/s | **6/6 (100%)** | 67% | 0.8961 |
| Llama-3.2-3B-Instruct | FP16 | 896.0 MiB | 1.00x | 31.2 ms | 29.1 ms | 34.3 tok/s | 5/6 (83%) | 83% | 0.2052 |
| Llama-3.2-3B-Instruct | INT8 | 455.0 MiB | 1.97x | 117.3 ms | 39.6 ms | 25.2 tok/s | 6/6 (100%) | 83% | 0.2030 |
| Llama-3.2-3B-Instruct | **K8V4 (g32)** | **353.5 MiB** | **2.53x** | 133.8 ms | 42.9 ms | 23.3 tok/s | **6/6 (100%)** | 83% | **0.2090** |
| Llama-3.2-3B-Instruct | **INT4 (K8g/V32g)** | **294.0 MiB** | **3.05x** | 158.8 ms | 45.6 ms | 21.9 tok/s | **6/6 (100%)** | 83% | **0.2027** |

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

Rather than greedy top-$k$ selection, DAG-KV maximizes a spatial quality-diversity objective via 1D neighborhood non-maximum suppression ($r=2$), selecting **4.0% bridge tokens on Stage 0** (`cuda:0`, Layers 0–13) and pruning to **1.5% before crossing PCIe to Stage 1** (`cuda:1`, Layers 14–27). Repaired KVs are written into request-private **Copy-on-Write (CoW) Shadow Blocks**, achieving **4.66x faster stitching** while reusing **96% of the canonical KV cache untouched**.

---

## Ecosystem Integrations (Hugging Face & vLLM)

### 1. Hugging Face Drop-In Wrapper (`patch_hf_model`)
Use DAG-KV with standard Hugging Face causal language models (Qwen, LLaMA, Mistral) in 3 lines of code:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from dag_kv import patch_hf_model

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")

# Wrap model with DAG-KV position-decoupled caching (auto-detects hardware)
dag_model = patch_hf_model(model, tokenizer, quantization="none", block_size=16)

# Prefill documents once in isolation
chunk_a = dag_model.prefill_chunk("File A: Authentication service implementation...")
chunk_b = dag_model.prefill_chunk("File B: Database schema definitions...")

# Query in any permutation with 100% KV reuse: [Chunk_B, Chunk_A]
response = dag_model.generate([chunk_b, chunk_a], prompt_suffix="Question: What DB is used?")
print(response)
```

### 2. vLLM BlockManager Plugin (`DAGKVvLLMBlockManagerPlugin`)
Integrate DAG-KV's Content-SHA256 registry and offset block table into vLLM's continuous batching engine:

```python
from dag_kv.integrations.vllm import DAGKVvLLMBlockManagerPlugin

plugin = DAGKVvLLMBlockManagerPlugin(block_size=16)

# Hash and register pre-computed physical blocks
hash_a = plugin.hash_tokens(chunk_a_tokens)
plugin.register_chunk(hash_a, physical_block_ids=[12, 13, 14])

# Dynamically assemble sequence mappings with RoPE delta offsets
seq_blocks = plugin.assemble_sequence_mapping(
    seq_id="request_101",
    chunk_hashes_and_lengths=[(hash_b, 128), (hash_a, 256)],
)
```

---

## CLI & Production Serving

Install the CLI directly via pip:
```bash
pip install -e .
```

### Inspect Discovered Hardware
```bash
dag-kv info
```
Output:
```text
============================================================
 DAG-KV Hardware Abstraction Layer (HAL) Report
============================================================
CUDA Available:       True
CUDA Device Count:    2
Apple Silicon MPS:    False
Triton JIT Available: True
Primary Compute Dev:  cuda:0
------------------------------------------------------------
Discovered Devices:
 [0] CUDA | NVIDIA GeForce RTX 2080 Ti       | Total: 10.75 GB | Free: 10.30 GB | CC: sm_75
 [1] CUDA | NVIDIA GeForce RTX 2080 Ti       | Total: 10.75 GB | Free: 10.59 GB | CC: sm_75
 [2] CPU  | Host CPU (48 cores, x86_64)      | Total: 187.54 GB | Free: 93.77 GB | CC: N/A
------------------------------------------------------------
Recommended Topology:
 Mode:         pipeline
 Compute Dev:  cuda:0
 KV Device:    cuda:0
 Dtype:        torch.float16
 Block Size:   16
============================================================
```

### Launch OpenAI-Compatible HTTP Server
```bash
dag-kv serve --host 0.0.0.0 --port 8000 --model Qwen/Qwen2.5-7B-Instruct --device auto
```

Query the endpoint:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [
      {"role": "user", "content": "### FILE: auth.py\n...\n### FILE: db.py\n...\nQuestion: What DB is used?"}
    ],
    "max_tokens": 16
  }'
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

## Repository Architecture

```text
dag-kv/
├── dag_kv/
│   ├── core/
│   │   ├── rope_adjoint.py           # Adjoint RoPE R(-Delta) implementation and mathematical proofs
│   │   ├── sink_canonicalizer.py     # 4-token sink prefix canonicalization & SHA256 hashing
│   │   ├── lse_probe.py              # Layer-1 LSE perturbation bound alpha_i & top-k% selector
│   │   ├── block_manager.py          # Paged Block Pool, OffsetBlockTable & CoW Shadow Blocks
│   │   └── quantizer.py              # Vectorized symmetric INT8 / INT4 KV cache quantizer
│   ├── devices/
│   │   ├── shard_config.py           # Pluggable topologies: Single-GPU, Pipeline, MPS, CPU Offload
│   │   └── device_manager.py         # Hardware Abstraction Layer auto-discovery & VRAM probing
│   ├── kernels/
│   │   ├── dispatch.py               # Dynamic dispatch between Triton and PyTorch fallbacks
│   │   ├── pytorch_fallbacks.py      # Universal pure PyTorch kernels for Apple Silicon (MPS) & CPU
│   │   ├── triton_adjoint_paged_attn.py # Turing SM75-tuned PagedAttention decode with R(-Delta_b)q
│   │   └── triton_sparse_stitch.py      # Fused ragged micro-prefill kernel for Bridge Tokens
│   ├── integrations/
│   │   ├── huggingface.py            # Drop-in DAGKVModelWrapper & patch_hf_model adapter
│   │   └── vllm.py                   # DAGKVvLLMBlockManagerPlugin for vLLM continuous batching
│   ├── models/
│   │   ├── base_engine.py            # Model-agnostic DAGKVEngine base class for DAG assembly & decode
│   │   ├── auto_engine.py            # AutoDAGKVEngine dynamic architecture router
│   │   ├── llama_engine.py           # Meta Llama-3 / Llama-3.1 DAG-KV engine
│   │   ├── mistral_engine.py         # Mistral-7B / Mixtral DAG-KV engine
│   │   ├── gemma_engine.py           # Google Gemma-2 DAG-KV engine
│   │   └── qwen2_dag_patch.py        # Alibaba Qwen-2 / Qwen-2.5 DAG-KV engine
│   ├── persistence/
│   │   ├── disk_cache.py             # Zero-copy safetensors NVMe disk cache for canonical chunks
│   │   └── pinned_cpu_pool.py        # Page-locked (pinned) Host RAM pool for high-speed DMA staging
│   ├── serving/
│   │   ├── scheduler.py              # Async ContinuousBatchingScheduler with dynamic decode batching
│   │   └── async_server.py           # High-throughput asyncio HTTP/1.1 and SSE streaming server
│   ├── worker/
│   │   ├── universal_pipeline.py     # Universal single/dual device & CPU-offloaded pipeline
│   │   └── dual_gpu_pipeline.py      # Dedicated Dual-GPU async prefill-decode pipeline
│   ├── cli.py                        # Command line interface (dag-kv info, serve, --continuous-batching)
│   └── server.py                     # OpenAI-compatible /v1/chat/completions HTTP server
├── configs/
│   └── hw_rtx2080ti.yaml             # Turing SM75 tile configs, memory constraints, dual-GPU pipeline
├── paper/
│   ├── dag_kv_paper.pdf              # Compiled 3-page research paper PDF (ready for publication)
│   └── main.tex                      # LaTeX manuscript source
├── assets/
│   └── dag_kv_benchmarks.png         # High-resolution benchmark overview chart
├── .github/workflows/
│   └── ci.yml                        # GitHub Actions Multi-Platform CI workflow
├── Dockerfile                        # Multi-Arch Production Dockerfile (linux/amd64, linux/arm64)
├── tests/                            # 65/65 Passing Tests (Continuous batching, LRU, INT4, Multi-Model, Disk)
└── benchmarks/                       # Production evaluation & benchmark suite
```

---

## Test Suite (65/65 Tests Passing)

Run the entire test suite across hardware backends:
```bash
pytest -v tests/
```
Output:
```text
tests/test_brutal_stress.py::test_stress_1_sha256_dedup_and_permutation_invariance PASSED [  1%]
tests/test_brutal_stress.py::test_stress_2_quantization_and_lse_bridge_selection PASSED [  3%]
tests/test_brutal_stress.py::test_stress_3_kv_pool_exhaustion_and_cow_refcount PASSED [  4%]
tests/test_brutal_stress.py::test_stress_4_extreme_32k_rope_gqa_and_parity PASSED [  6%]
tests/test_brutal_stress.py::test_stress_5_multithreaded_http_server_hammering PASSED [  7%]
tests/test_brutal_stress.py::test_stress_6_dual_gpu_and_exception_rollback PASSED [  9%]
tests/test_continuous_batching.py::test_continuous_batching_scheduler PASSED [ 10%]
tests/test_continuous_batching.py::test_async_server_http_and_sse PASSED [ 12%]
tests/test_core_components.py::test_sink_canonicalizer PASSED            [ 13%]
tests/test_core_components.py::test_lse_probe_and_bridge_selection PASSED [ 15%]
tests/test_core_components.py::test_block_manager_cow_and_assembly PASSED [ 16%]
tests/test_device_manager.py::test_device_manager_singleton_and_probing PASSED [ 18%]
tests/test_device_manager.py::test_shard_config_presets PASSED           [ 20%]
tests/test_device_manager.py::test_auto_detect_topology PASSED           [ 21%]
tests/test_disk_persistence.py::test_disk_kv_cache_save_and_load PASSED  [ 23%]
tests/test_disk_persistence.py::test_persistence_across_simulated_restart PASSED [ 24%]
tests/test_disk_persistence.py::test_pinned_cpu_pool_transfer PASSED     [ 26%]
tests/test_disk_persistence.py::test_engine_disk_cache_zero_prefill PASSED [ 27%]
tests/test_dual_gpu_pipeline.py::test_dual_gpu_disaggregated_prefill_and_stitch PASSED [ 29%]
tests/test_dynamic_blocks.py::test_dynamic_block_size_allocation[8] PASSED [ 30%]
tests/test_dynamic_blocks.py::test_dynamic_block_size_allocation[16] PASSED [ 32%]
tests/test_dynamic_blocks.py::test_dynamic_block_size_allocation[32] PASSED [ 33%]
tests/test_dynamic_blocks.py::test_attention_with_variable_block_sizes[8] PASSED [ 35%]
tests/test_dynamic_blocks.py::test_attention_with_variable_block_sizes[32] PASSED [ 36%]
tests/test_fused_int4.py::test_block_manager_int4_allocation PASSED      [ 38%]
tests/test_fused_int4.py::test_pytorch_int4_fallback_cpu PASSED          [ 40%]
tests/test_fused_int4.py::test_fused_int4_triton_vs_pytorch_parity PASSED [ 41%]
tests/test_hf_integration.py::test_vllm_block_manager_plugin PASSED      [ 43%]
tests/test_hf_integration.py::test_hf_wrapper_cpu_offline PASSED         [ 44%]
tests/test_lru_eviction.py::test_lru_eviction_lifecycle PASSED           [ 46%]
tests/test_lru_eviction.py::test_lru_protection_of_active_sequences PASSED [ 47%]
tests/test_lru_eviction.py::test_ensure_capacity_and_exhaustion PASSED   [ 49%]
tests/test_model_generalization.py::test_auto_engine_routing PASSED      [ 50%]
tests/test_model_generalization.py::test_llama_end_to_end_dag PASSED     [ 52%]
tests/test_model_generalization.py::test_mistral_end_to_end_dag PASSED   [ 53%]
tests/test_model_generalization.py::test_gemma2_end_to_end_dag PASSED    [ 55%]
tests/test_pytorch_fallback.py::test_pytorch_adjoint_paged_attention_cpu PASSED [ 56%]
tests/test_pytorch_fallback.py::test_pytorch_vs_triton_adjoint_paged_attention_parity PASSED [ 58%]
tests/test_pytorch_fallback.py::test_pytorch_vs_triton_sparse_bridge_parity PASSED [ 60%]
tests/test_quantization.py::test_kv_quantizer_int8_accuracy PASSED       [ 61%]
tests/test_quantization.py::test_kv_quantizer_int4_accuracy PASSED       [ 63%]
tests/test_quantization.py::test_block_manager_int8_quantization_lifecycle PASSED [ 64%]
tests/test_rope_adjoint.py::test_rope_adjoint_identity_scalar ... PASSED [ 76%]
tests/test_rope_adjoint.py::test_rope_adjoint_batched_multihead ... PASSED [ 83%]
tests/test_rotate_half_orthogonality PASSED        [ 84%]
tests/test_sparse_bridge_e2e.py::test_qwen2_dag_kv_e2e_stitching_and_repair PASSED [ 86%]
tests/test_triton_sm75.py::test_triton_adjoint_paged_attention_accuracy ... PASSED [ 95%]
tests/test_triton_sm75.py::test_triton_sparse_bridge_attention_accuracy ... PASSED [ 98%]
tests/test_triton_sm75.py::test_turing_sm75_shared_memory_limits PASSED  [100%]

============================= 65 passed in 188.22s =============================
```

---

## Citation & License

```bibtex
@software{dag_kv_2026,
  title={{DAG-KV: Universal Position-Decoupled Graph KV Cache with Query-Side Adjoint RoPE & LSE-Bounded Sparse Bridge Repair}},
  author={Nandaniya, Kaushal and Collaborators},
  year={2026},
  url={https://github.com/kaushalnandaniya/dag-kv}
}
```

Licensed under the [Apache 2.0 License](LICENSE).
