"""Llama-3 Architecture Adapter for DAG-KV Engine."""

from typing import List, Optional, Tuple
import torch
from .base_engine import DAGKVEngine


class LlamaDAGKVEngine(DAGKVEngine):
    """DAG-KV Engine tailored for Meta LLaMA 2 & LLaMA 3 families."""

    def _get_rope_base(self) -> float:
        return getattr(self.config, "rope_theta", 500000.0) or 500000.0

    def _get_sink_config(self) -> Tuple[str, Optional[List[int]]]:
        vocab_size = getattr(self.config, "vocab_size", 128256)
        # Standard LLaMA sink prefix IDs or clamped indices
        if hasattr(self.tokenizer, "encode"):
            try:
                ids = self.tokenizer.encode("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n", add_special_tokens=False)
                if len(ids) >= 4:
                    return "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n", ids[:4]
            except Exception:
                pass
        sink_ids = [min(i, vocab_size - 1) for i in range(4)]
        return "system\n\n", sink_ids
