"""Hugging Face Transformers Drop-In Model Wrapper & Integration.

Allows any Hugging Face CausalLM (Qwen, LLaMA, Mistral, Gemma) to transparently
execute with Position-Decoupled Graph KV Caching across any hardware.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from dag_kv.core.block_manager import BlockManager, OffsetBlockTable
from dag_kv.core.sink_canonicalizer import CanonicalChunk, SinkCanonicalizer
from dag_kv.devices.device_manager import get_device_manager
from dag_kv.devices.shard_config import QuantizationMode, ShardConfig
from dag_kv.models.qwen2_dag_patch import Qwen2DAGKVEngine


class DAGKVModelWrapper:
    """Drop-in wrapper around HuggingFace AutoModelForCausalLM."""

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        shard_config: Optional[ShardConfig] = None,
        max_num_blocks: int = 2048,
        block_size: int = 16,
        quantization: str = "none",
        bridge_top_k: float = 0.08,
    ):
        self.model = model
        self.tokenizer = tokenizer

        if shard_config is None:
            dev_mgr = get_device_manager()
            self.shard_config = dev_mgr.auto_detect_topology(block_size=block_size)
            try:
                p = next(model.parameters())
                self.shard_config.device = p.device
                self.shard_config.kv_device = p.device
                self.shard_config.prefill_device = p.device
                self.shard_config.dtype = p.dtype
            except StopIteration:
                pass
            self.shard_config.max_num_blocks = max_num_blocks
            self.shard_config.bridge_top_k = bridge_top_k
            if quantization != "none":
                self.shard_config.quantization = QuantizationMode(quantization)
        else:
            self.shard_config = shard_config

        self.engine = Qwen2DAGKVEngine(
            model=model,
            tokenizer=tokenizer,
            shard_config=self.shard_config,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: Optional[str] = None,
        quantization: str = "none",
        max_num_blocks: int = 2048,
        block_size: int = 16,
        bridge_top_k: float = 0.08,
        **kwargs,
    ) -> "DAGKVModelWrapper":
        """Instantiates model and tokenizer with automatic hardware routing."""
        dev_mgr = get_device_manager()
        target_device = dev_mgr.primary_device if (device is None or device == "auto") else torch.device(device)

        dtype = torch.float16 if target_device.type == "cuda" else torch.float32

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)
        device_map = None if target_device.type == "cpu" else target_device

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=dtype,
            device_map=device_map,
            **kwargs,
        )
        if target_device.type == "cpu":
            model = model.to("cpu")
        model.eval()

        shard_config = ShardConfig(
            device=target_device,
            dtype=dtype,
            block_size=block_size,
            max_num_blocks=max_num_blocks,
            quantization=QuantizationMode(quantization) if quantization != "none" else QuantizationMode.NONE,
            bridge_top_k=bridge_top_k,
        )

        return cls(
            model=model,
            tokenizer=tokenizer,
            shard_config=shard_config,
        )

    def prefill_chunk(
        self,
        text_or_tokens: Union[str, List[int], torch.Tensor],
        metadata: Optional[Dict] = None,
    ) -> CanonicalChunk:
        """Computes and caches position-canonical KV vectors for a document chunk."""
        return self.engine.prefill_canonical_chunk(text_or_tokens, metadata=metadata)

    def generate(
        self,
        chunks: List[CanonicalChunk],
        prompt_suffix: str = "",
        max_new_tokens: int = 32,
        temperature: float = 0.0,
    ) -> str:
        """Assembles DAG KV cache, performs bridge repair, and generates completion."""
        _, text = self.engine.generate(
            chunks=chunks,
            prompt_suffix=prompt_suffix,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        return text

    def chat(
        self,
        file_chunks: List[str],
        question: str,
        max_new_tokens: int = 64,
    ) -> str:
        """High-level multi-file repository Q&A interface."""
        cached_chunks = [self.prefill_chunk(chunk_text) for chunk_text in file_chunks]
        prompt = f"\n\nQuestion: {question}\nAnswer:"
        return self.generate(cached_chunks, prompt_suffix=prompt, max_new_tokens=max_new_tokens)


def patch_hf_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    **kwargs,
) -> DAGKVModelWrapper:
    """Drop-in monkeypatch / adapter turning any HuggingFace model into a DAG-KV engine."""
    return DAGKVModelWrapper(model, tokenizer, **kwargs)
