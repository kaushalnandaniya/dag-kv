"""Universal Hardware-Agnostic Execution Pipeline for DAG-KV.

Supports:
1. Single-Device (Single-GPU, Apple Silicon MPS, or CPU)
2. Dual-GPU Asynchronous Disaggregated Pipeline
3. CPU-Offloaded KV Pooling for VRAM-constrained machines
"""

from concurrent.futures import Future, ThreadPoolExecutor
import math
from typing import Dict, List, Optional, Tuple, Union
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from dag_kv.core.sink_canonicalizer import CanonicalChunk, SinkCanonicalizer
from dag_kv.devices.device_manager import get_device_manager
from dag_kv.devices.shard_config import ShardConfig, TopologyMode
from dag_kv.models.qwen2_dag_patch import Qwen2DAGKVEngine


class UniversalPipeline:
    """Universal pipeline handling single-device, multi-GPU pipeline, and CPU-offloaded inference."""

    def __init__(
        self,
        model_name_or_path: str = "Qwen/Qwen2.5-3B-Instruct",
        shard_config: Optional[ShardConfig] = None,
        max_num_blocks: int = 2048,
    ):
        self.model_name_or_path = model_name_or_path
        self.device_mgr = get_device_manager()

        if shard_config is None:
            self.shard_config = self.device_mgr.auto_detect_topology(
                preferred_topology=None,
                block_size=16,
            )
            self.shard_config.max_num_blocks = max_num_blocks
        else:
            self.shard_config = shard_config

        self.topology = self.shard_config.topology
        self.target_device = self.shard_config.device
        self.prefill_device = self.shard_config.prefill_device or self.target_device
        self.dtype = self.shard_config.dtype

        # Model and Tokenizer initialization
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        if self.topology == TopologyMode.PIPELINE and self.prefill_device != self.target_device:
            # Dual-device pipeline: load prefill model on prefill device, inference engine on target device
            self.prefill_model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                dtype=self.dtype,
                device_map=self.prefill_device,
            )
            self.prefill_model.eval()

            self.target_model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                dtype=self.dtype,
                device_map=self.target_device,
            )
            self.target_model.eval()

            self.engine = Qwen2DAGKVEngine(
                model=self.target_model,
                tokenizer=self.tokenizer,
                shard_config=self.shard_config,
            )
        else:
            # Single-device, CPU, or CPU-offload
            device_map = None if self.target_device.type == "cpu" else self.target_device
            self.target_model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                dtype=self.dtype,
                device_map=device_map,
            )
            if self.target_device.type == "cpu":
                self.target_model = self.target_model.to("cpu")
            self.target_model.eval()

            self.prefill_model = self.target_model
            self.engine = Qwen2DAGKVEngine(
                model=self.target_model,
                tokenizer=self.tokenizer,
                shard_config=self.shard_config,
            )

        self.canonicalizer = SinkCanonicalizer(
            sink_prefix="<|im_start|>system\n",
            num_sink_tokens=4,
        )

        # Streams and executor for async execution
        self.is_cuda = (self.target_device.type == "cuda")
        if self.is_cuda and torch.cuda.is_available():
            self.prefill_stream = torch.cuda.Stream(device=self.prefill_device)
            self.target_stream = torch.cuda.Stream(device=self.target_device)
        else:
            self.prefill_stream = None
            self.target_stream = None

        self.executor = ThreadPoolExecutor(max_workers=2)
        self._chunk_cache: Dict[str, CanonicalChunk] = {}

    def prefill_chunk_sync(
        self,
        content: Union[str, List[int], torch.Tensor],
        metadata: Optional[Dict] = None,
    ) -> CanonicalChunk:
        """Synchronously prefills a canonical chunk."""
        if self.topology == TopologyMode.PIPELINE and self.prefill_device != self.target_device:
            # Use dedicated prefill device and transfer
            return self._prefill_on_worker(content, metadata)
        else:
            return self.engine.prefill_canonical_chunk(content, metadata=metadata)

    def prefill_chunk_async(
        self,
        content: Union[str, List[int], torch.Tensor],
        metadata: Optional[Dict] = None,
    ) -> Future:
        """Asynchronously dispatches prefill to worker thread."""
        return self.executor.submit(self.prefill_chunk_sync, content, metadata)

    def _prefill_on_worker(
        self,
        text_or_tokens: Union[str, List[int], torch.Tensor],
        metadata: Optional[Dict] = None,
    ) -> CanonicalChunk:
        """Prefills on prefill_device and transfers to target_device."""
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

        stream_ctx = torch.cuda.stream(self.prefill_stream) if self.prefill_stream else torch.no_grad()
        with stream_ctx:
            input_ids, attention_mask, position_ids = self.canonicalizer.prepare_prefill_input(
                token_ids, device=self.prefill_device
            )

            outputs = self.prefill_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=True,
                logits_to_keep=1,
            )

            pkv = outputs.past_key_values
            total_len = input_ids.shape[-1]
            num_layers = self.engine.num_layers
            num_heads = self.engine.num_heads
            num_kv_heads = self.engine.num_kv_heads
            gqa_ratio = num_heads // num_kv_heads
            sm_scale = self.engine.sm_scale

            layer_k, layer_v, layer_lse = [], [], []
            causal_mask = torch.triu(
                torch.full((total_len, total_len), float("-inf"), device=self.prefill_device), diagonal=1
            ).unsqueeze(0)

            for l_idx in range(num_layers):
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
                    scores = torch.matmul(k_exp.to(torch.float32), k_exp.to(torch.float32).transpose(-1, -2)) * sm_scale
                    lse_l = torch.logsumexp(scores + causal_mask, dim=-1)
                else:
                    lse_l = torch.zeros((num_heads, total_len), dtype=torch.float32, device=self.prefill_device)
                layer_lse.append(lse_l)

            all_k = torch.stack(layer_k, dim=0)
            all_v = torch.stack(layer_v, dim=0)
            all_lse = torch.stack(layer_lse, dim=0)

        if self.prefill_stream:
            self.prefill_stream.synchronize()

        # Transfer to target device (using host pinned memory staging if non-P2P CUDA)
        target_k = all_k.to(self.target_device, dtype=self.dtype)
        target_v = all_v.to(self.target_device, dtype=self.dtype)
        target_lse = all_lse.to(self.target_device)
        target_tokens = token_ids.to(self.target_device)

        chunk = self.canonicalizer.canonicalize_kv(
            full_k=target_k,
            full_v=target_v,
            full_lse=target_lse,
            chunk_token_ids=target_tokens,
            metadata=metadata,
        )
        self.engine.block_manager.register_canonical_chunk(chunk)
        self._chunk_cache[chunk_hash] = chunk
        return chunk

    def stitch_and_decode(
        self,
        chunks: List[CanonicalChunk],
        query: str = "",
        max_new_tokens: int = 16,
    ) -> Tuple[List[int], str]:
        """Stitches canonical chunks, repairs bridge tokens, and decodes."""
        return self.engine.generate(
            chunks=chunks,
            prompt_suffix=query,
            max_new_tokens=max_new_tokens,
        )
