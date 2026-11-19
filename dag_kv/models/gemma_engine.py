"""Gemma & Gemma-2 Architecture Adapter for DAG-KV Engine."""

import math
from typing import List, Optional, Tuple
import torch
from .base_engine import DAGKVEngine


class Gemma2DAGKVEngine(DAGKVEngine):
    """DAG-KV Engine tailored for Google Gemma & Gemma-2 models."""

    def _get_rope_base(self) -> float:
        return getattr(self.config, "rope_theta", 10000.0) or 10000.0

    def _get_head_dim(self) -> int:
        return getattr(self.config, "head_dim", 256)

    def _get_sm_scale(self) -> float:
        scalar = getattr(self.config, "query_pre_attn_scalar", None)
        if scalar is not None and scalar > 0:
            return float(scalar) ** -0.5
        return 1.0 / math.sqrt(self.head_dim)

    def get_lm_head(self, h: torch.Tensor) -> torch.Tensor:
        logits = self.model.lm_head(h)
        cap = getattr(self.config, "final_logit_softcapping", None)
        if cap is not None and cap > 0:
            logits = torch.tanh(logits / cap) * cap
        return logits

    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Gemma multiplies input embeddings by sqrt(hidden_size)."""
        emb = self.model.get_input_embeddings()(token_ids).to(self.dtype)
        normalizer = torch.tensor(math.sqrt(self.config.hidden_size), dtype=self.dtype, device=self.device)
        return emb * normalizer

    def apply_attn_output_and_mlp(self, layer, res: torch.Tensor, attn_out: torch.Tensor) -> torch.Tensor:
        """Applies Gemma-2 post-attention norm, pre-feedforward, and post-feedforward norms."""
        o_proj = layer.self_attn.o_proj(attn_out)
        if hasattr(layer, "post_attention_layernorm"):
            o_proj = layer.post_attention_layernorm(o_proj)
        h = res + o_proj

        mlp_res = h
        if hasattr(layer, "pre_feedforward_layernorm"):
            h = layer.pre_feedforward_layernorm(h)
        mlp_out = layer.mlp(h)
        if hasattr(layer, "post_feedforward_layernorm"):
            mlp_out = layer.post_feedforward_layernorm(mlp_out)
        return mlp_res + mlp_out

    def _get_sink_config(self) -> Tuple[str, Optional[List[int]]]:
        vocab_size = getattr(self.config, "vocab_size", 256000)
        sink_ids = [min(i, vocab_size - 1) for i in range(4)]
        return "<start_of_turn>user\n", sink_ids
