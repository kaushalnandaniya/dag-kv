"""Log-Sum-Exp (LSE) Perturbation Bound & Sparse Bridge Token Selector (Pillar 3).

Mathematical Theorem (Online Softmax Decomposition & Perturbation Bound):
When stitching context chunks [Doc_B, Doc_A], tokens in Doc_A missed attending to Doc_B.
Let l_i^local be the cached local LSE scalar for token i:
    l_i^local = log sum_{j in local} exp(s_ij)

Let l_{i -> prior} be the cross-attention LSE from token i to all preceding chunks:
    l_{i -> prior} = log sum_{k in prior} exp(s_ik)

The exact convex mixing weight alpha_i in [0, 1] is:
    alpha_i = exp(l_{i -> prior}) / (exp(l_i^local) + exp(l_{i -> prior}))
            = sigmoid(l_{i -> prior} - l_i^local)

Output drift is strictly bounded:
    || o_i^exact - o_i^cached ||_2 <= 2 * alpha_i * max_j || v_j ||_2

Tokens with alpha_i close to 0 suffer virtually zero drift and need no recomputation.
We compute alpha_i at Layer 1 and select the top k% (default 8%) Bridge Tokens
for sparse micro-prefill repair.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F


@dataclass
class BridgeSelectionResult:
    """Result of LSE probe and Bridge Token selection."""
    bridge_indices: torch.Tensor       # 1D int64 indices of selected tokens within the target chunk
    global_bridge_indices: torch.Tensor # 1D int64 global position indices
    alpha_scores: torch.Tensor         # Per-token mean alpha scores (L_target,)
    top_k_percent: float               # Percentage of tokens selected (e.g. 0.08)
    num_bridge_tokens: int             # Count of bridge tokens
    chunk_length: int                  # Total length of target chunk


def compute_lse_perturbation(
    q_layer1: torch.Tensor,
    k_prior: torch.Tensor,
    lse_local: torch.Tensor,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Computes per-head and mean alpha_i mixing weights for target tokens attending to prior K.

    Args:
        q_layer1: Queries for target tokens. Shape (num_heads, L_target, head_dim)
        k_prior: Keys for prior tokens. Shape (num_kv_heads, L_prior, head_dim)
        lse_local: Cached local LSE scalars for target tokens. Shape (num_heads, L_target)
        scale: Softmax scale 1 / sqrt(head_dim). Defaults to 1 / sqrt(dim).

    Returns:
        alpha: (L_target,) mean mixing weight across attention heads in [0, 1].
    """
    num_heads, l_target, head_dim = q_layer1.shape
    num_kv_heads, l_prior, _ = k_prior.shape

    if scale is None:
        scale = 1.0 / (head_dim ** 0.5)

    # Handle Grouped Query Attention (GQA) if num_heads != num_kv_heads
    if num_heads != num_kv_heads:
        group_size = num_heads // num_kv_heads
        k_prior = k_prior.repeat_interleave(group_size, dim=0)

    # Compute dot products: (num_heads, L_target, L_prior)
    # Using FP32 accumulation for numerical precision on sm_75
    scores = torch.matmul(q_layer1.to(torch.float32), k_prior.to(torch.float32).transpose(-1, -2)) * scale

    # Cross-attention LSE: logsumexp over prior tokens (dim=-1)
    lse_prior = torch.logsumexp(scores, dim=-1)  # (num_heads, L_target)

    # Exact sigmoid decomposition: alpha_i = sigmoid(l_{i -> prior} - l_i^local)
    # lse_local is (num_heads, L_target)
    lse_diff = lse_prior - lse_local.to(torch.float32)
    alpha_per_head = torch.sigmoid(lse_diff)  # (num_heads, L_target)

    # Aggregate across heads: mean or max. Mean gives average expected drift.
    alpha_mean = torch.mean(alpha_per_head, dim=0)  # (L_target,)
    return alpha_mean


def select_bridge_tokens(
    alpha: torch.Tensor,
    top_k_percent: float = 0.08,
    min_tokens: int = 1,
    max_tokens: Optional[int] = None,
    chunk_global_offset: int = 0,
) -> BridgeSelectionResult:
    """Selects the top k% Bridge Tokens based on alpha scores.

    Args:
        alpha: (L_target,) mixing weight tensor in [0, 1]
        top_k_percent: fraction of tokens to select (e.g. 0.08 for 8%)
        min_tokens: minimum number of bridge tokens to select
        max_tokens: optional upper bound on bridge tokens
        chunk_global_offset: global prompt start position of this chunk

    Returns:
        BridgeSelectionResult with local and global indices
    """
    length = alpha.shape[0]
    num_to_select = max(min_tokens, int(round(length * top_k_percent)))
    if max_tokens is not None:
        num_to_select = min(num_to_select, max_tokens)
    num_to_select = min(num_to_select, length)

    # Top-k selection
    _, top_indices = torch.topk(alpha, k=num_to_select, largest=True, sorted=False)
    # Sort indices in ascending sequence order for causal prefix preservation
    bridge_indices, _ = torch.sort(top_indices)

    global_bridge_indices = bridge_indices + chunk_global_offset

    return BridgeSelectionResult(
        bridge_indices=bridge_indices,
        global_bridge_indices=global_bridge_indices,
        alpha_scores=alpha,
        top_k_percent=top_k_percent,
        num_bridge_tokens=num_to_select,
        chunk_length=length,
    )


class LSEProbe:
    """Layer-1 LSE Probe Module for DAG-KV stitching."""

    def __init__(self, top_k_percent: float = 0.08, probe_layer: int = 1):
        self.top_k_percent = top_k_percent
        self.probe_layer = probe_layer

    def probe_and_select(
        self,
        q_probe: torch.Tensor,
        k_prior: torch.Tensor,
        lse_local: torch.Tensor,
        chunk_global_offset: int,
    ) -> BridgeSelectionResult:
        """Runs the probe and selects bridge tokens for the target chunk."""
        alpha = compute_lse_perturbation(q_probe, k_prior, lse_local)
        return select_bridge_tokens(
            alpha=alpha,
            top_k_percent=self.top_k_percent,
            chunk_global_offset=chunk_global_offset,
        )
