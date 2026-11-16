"""Dual-GPU Asynchronous Disaggregated Pipeline Architecture (2x RTX 2080 Ti sm_75).

Topology-Aware Design (NUMA SYS Interconnect / P2P=False):
- GPU 1 (`cuda:1`, NUMA 1): Asynchronous Canonical Chunk Prefill Worker.
  Computes position-zero canonical KV caches + exact Layer-1 causal LSE with the
  4-token sink prefix on a dedicated CUDA stream, then stages tensors through
  page-locked (pinned) Host RAM (`pin_memory=True`) to avoid cross-socket P2P stalls.
- GPU 0 (`cuda:0`, NUMA 0): Stitched DAG-KV Inference Engine.
  Asynchronously ingests canonical chunks into its HBM BlockManager on a dedicated
  DMA stream while executing Layer-1 LSE Bridge Selection, Fused Triton Sparse Stitch,
  and Adjoint-RoPE PagedAttention Decode on the compute stream.
"""

from concurrent.futures import Future, ThreadPoolExecutor
import math
from typing import Dict, List, Optional, Tuple, Union
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dag_kv.core.sink_canonicalizer import CanonicalChunk, SinkCanonicalizer
from dag_kv.models.qwen2_dag_patch import Qwen2DAGKVEngine


class AsyncCanonicalChunkPrefiller:
    """Asynchronous worker running on GPU 1 to prefill and canonicalize chunks."""

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-3B-Instruct",
        prefill_device: str = "cuda:1",
        target_device: str = "cuda:0",
    ):
        self.prefill_device = torch.device(prefill_device)
        self.target_device = torch.device(target_device)
        self.can_p2p = (
            self.prefill_device.type == "cuda"
            and self.target_device.type == "cuda"
            and self.prefill_device.index != self.target_device.index
            and torch.cuda.can_device_access_peer(self.prefill_device.index, self.target_device.index)
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=torch.float16,
            device_map=self.prefill_device,
        )
        self.model.eval()

        self.config = self.model.config
        self.num_layers = self.config.num_hidden_layers
        self.num_heads = self.config.num_attention_heads
        self.num_kv_heads = self.config.num_key_value_heads
        self.head_dim = getattr(self.config, "head_dim", self.config.hidden_size // self.num_heads)
        self.sm_scale = 1.0 / math.sqrt(self.head_dim)

        self.canonicalizer = SinkCanonicalizer(
            sink_prefix="<|im_start|>system\n",
            num_sink_tokens=4,
        )
        self.prefill_stream = torch.cuda.Stream(device=self.prefill_device)
        self.dma_stream_target = torch.cuda.Stream(device=self.target_device)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self._chunk_cache: Dict[str, CanonicalChunk] = {}

    @torch.inference_mode()
    def prefill_chunk(
        self,
        text_or_tokens: Union[str, List[int], torch.Tensor],
        metadata: Optional[Dict] = None,
    ) -> CanonicalChunk:
        """Prefills chunk on GPU 1 and transfers to GPU 0 via Pinned Host DMA (or P2P)."""
        if isinstance(text_or_tokens, str):
            token_ids = self.tokenizer.encode(text_or_tokens, add_special_tokens=False)
            token_ids = torch.tensor(token_ids, dtype=torch.int64, device=self.prefill_device)
        elif isinstance(text_or_tokens, list):
            token_ids = torch.tensor(text_or_tokens, dtype=torch.int64, device=self.prefill_device)
        else:
            token_ids = text_or_tokens.to(self.prefill_device, dtype=torch.int64)

        if token_ids.dim() > 1:
            token_ids = token_ids.squeeze()

        chunk_hash = self.canonicalizer.compute_chunk_hash(token_ids)
        if chunk_hash in self._chunk_cache:
            return self._chunk_cache[chunk_hash]

        with torch.cuda.stream(self.prefill_stream):
            input_ids, attention_mask, position_ids = self.canonicalizer.prepare_prefill_input(
                token_ids, device=self.prefill_device
            )

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
                logits_to_keep=1,
            )

            pkv = outputs.past_key_values
            total_len = input_ids.shape[-1]
            gqa_ratio = self.num_heads // self.num_kv_heads

            layer_k, layer_v, layer_lse = [], [], []
            causal_mask = torch.triu(
                torch.full((total_len, total_len), float("-inf"), device=self.prefill_device), diagonal=1
            ).unsqueeze(0)

            for l_idx in range(self.num_layers):
                if hasattr(pkv, "layers"):
                    k_l = pkv.layers[l_idx].keys.squeeze(0)
                    v_l = pkv.layers[l_idx].values.squeeze(0)
                elif hasattr(pkv, "key_cache"):
                    k_l = pkv.key_cache[l_idx].squeeze(0)
                    v_l = pkv.value_cache[l_idx].squeeze(0)
                else:
                    k_l = pkv[l_idx][0].squeeze(0)
                    v_l = pkv[l_idx][1].squeeze(0)

                layer_k.append(k_l)
                layer_v.append(v_l)

                if l_idx == 1:
                    k_exp = k_l.repeat_interleave(gqa_ratio, dim=0) if gqa_ratio > 1 else k_l
                    scores = torch.matmul(k_exp.to(torch.float32), k_exp.to(torch.float32).transpose(-1, -2)) * self.sm_scale
                    lse_l = torch.logsumexp(scores + causal_mask, dim=-1)
                else:
                    lse_l = torch.zeros((self.num_heads, total_len), dtype=torch.float32, device=self.prefill_device)
                layer_lse.append(lse_l)

            all_k = torch.stack(layer_k, dim=0)
            all_v = torch.stack(layer_v, dim=0)
            all_lse = torch.stack(layer_lse, dim=0)

            chunk_gpu1 = self.canonicalizer.canonicalize_kv(
                full_k=all_k,
                full_v=all_v,
                full_lse=all_lse,
                chunk_token_ids=token_ids,
                metadata=metadata,
            )

            if self.can_p2p or self.prefill_device == self.target_device:
                chunk_target = chunk_gpu1.to(self.target_device, non_blocking=True)
                self.prefill_stream.synchronize()
            else:
                # Stage through Page-Locked (Pinned) Host Memory to avoid non-P2P cross-NUMA stall
                k_pin = torch.empty_like(chunk_gpu1.k_cache, device="cpu", pin_memory=True)
                v_pin = torch.empty_like(chunk_gpu1.v_cache, device="cpu", pin_memory=True)
                lse_pin = torch.empty_like(chunk_gpu1.lse_local, device="cpu", pin_memory=True)

                k_pin.copy_(chunk_gpu1.k_cache, non_blocking=True)
                v_pin.copy_(chunk_gpu1.v_cache, non_blocking=True)
                lse_pin.copy_(chunk_gpu1.lse_local, non_blocking=True)

                d2h_event = torch.cuda.Event()
                d2h_event.record(self.prefill_stream)
                d2h_event.synchronize()

                with torch.cuda.stream(self.dma_stream_target):
                    k_t = k_pin.to(self.target_device, non_blocking=True)
                    v_t = v_pin.to(self.target_device, non_blocking=True)
                    lse_t = lse_pin.to(self.target_device, non_blocking=True)
                    canon_pos_t = chunk_gpu1.canonical_positions.to(self.target_device, non_blocking=True)

                self.dma_stream_target.synchronize()
                chunk_target = CanonicalChunk(
                    chunk_id=chunk_gpu1.chunk_id,
                    token_ids=chunk_gpu1.token_ids,
                    k_cache=k_t,
                    v_cache=v_t,
                    lse_local=lse_t,
                    canonical_positions=canon_pos_t,
                    length=chunk_gpu1.length,
                    metadata=chunk_gpu1.metadata,
                )

        self._chunk_cache[chunk_hash] = chunk_target
        return chunk_target

    def submit_prefill_async(
        self,
        text_or_tokens: Union[str, List[int], torch.Tensor],
        metadata: Optional[Dict] = None,
    ) -> Future:
        """Submits an asynchronous prefill job to GPU 1's background thread."""
        return self.executor.submit(self.prefill_chunk, text_or_tokens, metadata)


class DualGPUPipeline:
    """Coordinates Async GPU 1 Prefilling with GPU 0 Stitched DAG Inference."""

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-3B-Instruct",
        inference_device: str = "cuda:0",
        prefill_device: Optional[str] = None,
        max_num_blocks: int = 2048,
        bridge_top_k: float = 0.08,
    ):
        num_gpus = torch.cuda.device_count()
        if prefill_device is None:
            prefill_device = "cuda:1" if num_gpus > 1 else "cuda:0"

        self.inference_device = inference_device
        self.prefill_device = prefill_device
        self.is_dual_gpu = (inference_device != prefill_device)

        print(f"Initializing DAG-KV Pipeline: Inference on {inference_device}, Prefill on {prefill_device}")

        # 1. Initialize GPU 0 Stitched Inference Engine
        self.engine = Qwen2DAGKVEngine.from_pretrained(
            model_name_or_path=model_name_or_path,
            device=inference_device,
            max_num_blocks=max_num_blocks,
            bridge_top_k=bridge_top_k,
        )

        # Seed GPU 0's 4-token Canonical Sink Block at physical block 0 once at startup
        self._seed_engine_sink_block()

        # 2. Initialize GPU 1 Disaggregated Prefill Worker
        if self.is_dual_gpu:
            self.prefiller = AsyncCanonicalChunkPrefiller(
                model_name_or_path=model_name_or_path,
                prefill_device=prefill_device,
                target_device=inference_device,
            )
        else:
            self.prefiller = None

    @torch.inference_mode()
    def _seed_engine_sink_block(self):
        """Ensures GPU 0 has the 4-token canonical sink block allocated at positions [0..3]."""
        if self.engine._sink_phys_block is None:
            dummy_ids = torch.tensor([198], dtype=torch.int64, device=self.engine.device)
            _ = self.engine.prefill_canonical_chunk(dummy_ids)

    def register_transferred_chunk(self, chunk: CanonicalChunk) -> CanonicalChunk:
        """Registers a GPU 1 prefilled chunk into GPU 0's BlockManager."""
        if chunk.chunk_id not in self.engine._chunk_cache:
            self.engine.block_manager.register_canonical_chunk(chunk)
            self.engine._chunk_cache[chunk.chunk_id] = chunk
        return chunk

    def prefill_chunk(
        self,
        content: Union[str, List[int], torch.Tensor],
        metadata: Optional[Dict] = None,
    ) -> CanonicalChunk:
        """Prefills chunk on GPU 1 (if dual-GPU) and registers it into GPU 0's BlockManager."""
        if self.is_dual_gpu and self.prefiller is not None:
            chunk = self.prefiller.prefill_chunk(content, metadata)
            return self.register_transferred_chunk(chunk)
        else:
            return self.engine.prefill_canonical_chunk(content, metadata)

    def prefill_chunk_async(
        self,
        content: Union[str, List[int], torch.Tensor],
        metadata: Optional[Dict] = None,
    ) -> Future:
        """Launches non-blocking chunk prefill on GPU 1 while GPU 0 continues serving."""
        if self.is_dual_gpu and self.prefiller is not None:
            return self.prefiller.submit_prefill_async(content, metadata)
        else:
            fut: Future = Future()
            fut.set_result(self.engine.prefill_canonical_chunk(content, metadata))
            return fut

    def stitch_and_generate(
        self,
        chunks: List[CanonicalChunk],
        prompt_suffix: str = "",
        max_new_tokens: int = 16,
        temperature: float = 0.0,
    ) -> Tuple[List[int], str]:
        """Registers any pending chunks and runs stitched generation on GPU 0."""
        for c in chunks:
            self.register_transferred_chunk(c)
        return self.engine.generate(
            chunks=chunks,
            prompt_suffix=prompt_suffix,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
