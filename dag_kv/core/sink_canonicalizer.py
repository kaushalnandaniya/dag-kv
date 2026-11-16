"""Canonical Sink Prefix & KV Extraction (Pillar 2).

Solves Position-Zero Norm Pathology (StreamingLLM / Attention Sinks):
LLMs develop massive activation norms on the first 1-4 tokens at position 0.
Every chunk is pre-filled with a fixed 4-token canonical sink prefix:
    SINK_PREFIX = "<|im_start|>system\n" (padded or mapped to 4 tokens)
placing chunk tokens at canonical positions [4, ..., L_c + 3].
After pre-fill, the 4 sink slots are stripped, preserving the clean canonical
KV vectors and per-token local Log-Sum-Exp scalars.
"""

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import torch


# Default 4-token canonical sink prefix IDs for Qwen2.5 tokenizer:
# "<|im_start|>system\n" -> [151644, 8948, 198] + newline [198] = 4 tokens
DEFAULT_QWEN_SINK_IDS = [151644, 8948, 198, 198]


@dataclass
class CanonicalChunk:
    """Stored representation of an isolated canonical document chunk."""
    chunk_id: str                      # SHA256 content hash of token_ids
    token_ids: torch.Tensor             # (L_c,) int64
    k_cache: torch.Tensor              # (num_layers, num_kv_heads, L_c, head_dim) FP16
    v_cache: torch.Tensor              # (num_layers, num_kv_heads, L_c, head_dim) FP16
    lse_local: torch.Tensor            # (num_layers, num_heads, L_c) FP32
    canonical_positions: torch.Tensor  # (L_c,) int64: [num_sink, ..., num_sink + L_c - 1]
    length: int
    metadata: Dict = field(default_factory=dict)

    def to(self, device: Union[str, torch.device], non_blocking: bool = True) -> "CanonicalChunk":
        """Move chunk tensors to target device asynchronously."""
        return CanonicalChunk(
            chunk_id=self.chunk_id,
            token_ids=self.token_ids.to(device, non_blocking=non_blocking),
            k_cache=self.k_cache.to(device, non_blocking=non_blocking),
            v_cache=self.v_cache.to(device, non_blocking=non_blocking),
            lse_local=self.lse_local.to(device, non_blocking=non_blocking),
            canonical_positions=self.canonical_positions.to(device, non_blocking=non_blocking),
            length=self.length,
            metadata=self.metadata,
        )


class SinkCanonicalizer:
    """Manages sink token injection and post-prefill KV stripping."""

    def __init__(
        self,
        sink_prefix: str = "<|im_start|>system\n",
        num_sink_tokens: int = 4,
        sink_token_ids: Optional[List[int]] = None,
    ):
        self.sink_prefix = sink_prefix
        self.num_sink_tokens = num_sink_tokens
        if sink_token_ids is not None:
            self.sink_token_ids = list(sink_token_ids)[:num_sink_tokens]
            while len(self.sink_token_ids) < num_sink_tokens:
                self.sink_token_ids.append(198)  # pad with newline
        else:
            self.sink_token_ids = DEFAULT_QWEN_SINK_IDS[:num_sink_tokens]

    @staticmethod
    def compute_chunk_hash(token_ids: Union[torch.Tensor, List[int]]) -> str:
        """Computes content SHA256 of token sequence."""
        if isinstance(token_ids, torch.Tensor):
            data = token_ids.detach().cpu().to(torch.int64).numpy().tobytes()
        else:
            data = torch.tensor(token_ids, dtype=torch.int64).numpy().tobytes()
        return hashlib.sha256(data).hexdigest()

    def prepare_prefill_input(
        self,
        chunk_token_ids: Union[torch.Tensor, List[int]],
        device: Optional[torch.device] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepends sink tokens to chunk tokens for isolated prefill.

        Returns:
            (input_ids, attention_mask, position_ids)
            where position_ids start at 0 and chunk tokens sit at [num_sink, ...].
        """
        if not isinstance(chunk_token_ids, torch.Tensor):
            chunk_tokens = torch.tensor(chunk_token_ids, dtype=torch.int64, device=device)
        else:
            chunk_tokens = chunk_token_ids.to(dtype=torch.int64, device=device)

        if chunk_tokens.dim() == 1:
            chunk_tokens = chunk_tokens.unsqueeze(0)

        batch_size = chunk_tokens.shape[0]
        sink_tensor = torch.tensor(
            self.sink_token_ids, dtype=torch.int64, device=chunk_tokens.device
        ).unsqueeze(0).expand(batch_size, -1)

        input_ids = torch.cat([sink_tensor, chunk_tokens], dim=-1)
        total_len = input_ids.shape[-1]
        attention_mask = torch.ones((batch_size, total_len), dtype=torch.long, device=input_ids.device)
        position_ids = torch.arange(total_len, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        return input_ids, attention_mask, position_ids

    def canonicalize_kv(
        self,
        full_k: torch.Tensor,
        full_v: torch.Tensor,
        full_lse: torch.Tensor,
        chunk_token_ids: torch.Tensor,
        metadata: Optional[Dict] = None,
    ) -> CanonicalChunk:
        """Strips the 4 sink positions and packages the canonical chunk.

        Args:
            full_k: (num_layers, num_kv_heads, total_len, head_dim) or (layers, batch, kv_heads, seq_len, head_dim)
            full_v: same shape as full_k
            full_lse: (num_layers, num_heads, total_len) or (layers, batch, heads, seq_len)
            chunk_token_ids: (L_c,) token ids of the chunk
            metadata: optional dict
        """
        s = self.num_sink_tokens
        
        # Normalize shapes: ensure seq_len is along dim -2
        k_clean = full_k[..., s:, :]
        v_clean = full_v[..., s:, :]
        lse_clean = full_lse[..., s:]

        # Squeeze batch dimension if present and batch_size == 1
        if k_clean.dim() == 5 and k_clean.shape[1] == 1:
            k_clean = k_clean.squeeze(1)
            v_clean = v_clean.squeeze(1)
            lse_clean = lse_clean.squeeze(1)

        # Ensure float16 for KV, float32 for LSE
        k_clean = k_clean.to(dtype=torch.float16)
        v_clean = v_clean.to(dtype=torch.float16)
        lse_clean = lse_clean.to(dtype=torch.float32)

        raw_ids = chunk_token_ids.squeeze() if chunk_token_ids.dim() > 1 else chunk_token_ids
        length = raw_ids.shape[0]
        canonical_positions = torch.arange(s, s + length, dtype=torch.long, device=raw_ids.device)
        chunk_id = self.compute_chunk_hash(raw_ids)

        return CanonicalChunk(
            chunk_id=chunk_id,
            token_ids=raw_ids.detach().cpu(),
            k_cache=k_clean,
            v_cache=v_clean,
            lse_local=lse_clean,
            canonical_positions=canonical_positions,
            length=length,
            metadata=metadata or {},
        )
