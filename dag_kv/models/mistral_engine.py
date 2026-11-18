"""Mistral & Mixtral Architecture Adapter for DAG-KV Engine."""

from typing import List, Optional, Tuple
import torch
from .base_engine import DAGKVEngine


class MistralDAGKVEngine(DAGKVEngine):
    """DAG-KV Engine tailored for Mistral-7B and Mixtral MoE architectures."""

    def _get_rope_base(self) -> float:
        return getattr(self.config, "rope_theta", 1000000.0) or 1000000.0

    def _get_sink_config(self) -> Tuple[str, Optional[List[int]]]:
        vocab_size = getattr(self.config, "vocab_size", 32000)
        if hasattr(self.tokenizer, "encode"):
            try:
                ids = self.tokenizer.encode("[INST] system\n", add_special_tokens=False)
                if len(ids) >= 4:
                    return "[INST] system\n", ids[:4]
            except Exception:
                pass
        sink_ids = [min(i, vocab_size - 1) for i in range(4)]
        return "[INST] system\n", sink_ids
