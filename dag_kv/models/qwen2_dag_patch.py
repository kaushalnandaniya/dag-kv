"""Qwen2 Model Integration with DAG-KV Engine - Subclasses DAGKVEngine."""

from typing import Dict, List, Optional, Tuple, Union
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from dag_kv.devices.shard_config import ShardConfig
from .base_engine import DAGKVEngine


class Qwen2DAGKVEngine(DAGKVEngine):
    """End-to-end inference and stitching engine for Qwen2 on DAG-KV."""

    def _get_rope_base(self) -> float:
        return getattr(self.config, "rope_theta", 1000000.0) or 1000000.0

    def _get_sink_config(self) -> Tuple[str, Optional[List[int]]]:
        vocab_size = getattr(self.config, "vocab_size", 160000)
        sink_token_ids = None
        if vocab_size <= 151644:
            sink_token_ids = [min(i, vocab_size - 1) for i in range(4)]
        return "<|im_start|>system\n", sink_token_ids

    @classmethod
    def from_pretrained_sharded(
        cls,
        model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct",
        max_num_blocks: int = 1024,
        bridge_top_k: float = 0.08,
    ) -> "Qwen2DAGKVEngine":
        """Loads a large model sharded across GPUs in FP16."""
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=torch.float16,
            device_map="auto",
        )
        model.eval()
        return cls(
            model=model,
            tokenizer=tokenizer,
            max_num_blocks=max_num_blocks,
            device="cuda:0",
            bridge_top_k=bridge_top_k,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "Qwen/Qwen2.5-3B-Instruct",
        device: str = "cuda:0",
        max_num_blocks: int = 2048,
        bridge_top_k: float = 0.08,
    ) -> "Qwen2DAGKVEngine":
        """Loads a model onto a single device in FP16."""
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=torch.float16,
            device_map=device,
        )
        model.eval()
        return cls(
            model=model,
            tokenizer=tokenizer,
            max_num_blocks=max_num_blocks,
            device=device,
            bridge_top_k=bridge_top_k,
        )
