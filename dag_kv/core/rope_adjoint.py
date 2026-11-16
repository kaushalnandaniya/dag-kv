"""Query-Side Adjoint RoPE Reference and Vectorized Mathematics.

Mathematical Foundation:
Standard Rotary Position Embedding (RoPE) maps key and query vectors into SO(2)
subspaces. In half-split RoPE (Qwen2, LLaMA), for hidden dimension d and head_dim D:
    R(m) x = x * cos(m * theta) + rotate_half(x) * sin(m * theta)

When a document chunk c with canonical local positions p in [0, L_c - 1] is placed
at global offset Delta_c, its tokens sit at global positions m = p + Delta_c.
Instead of mutating cached keys R(p)k in HBM, we exploit the orthogonal adjoint
property of SO(2):
    R(p + Delta_c) = R(Delta_c) R(p)
    R(Delta_c)^T = R(-Delta_c)

Therefore:
    <R(n)q, R(p + Delta_c)k> = <R(-Delta_c) (R(n)q), R(p)k>
"""

import math
from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dimensions of the input tensor.

    Compatible with HuggingFace Transformers (Qwen2, Llama).
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def compute_inv_freq(dim: int, base: float = 10000.0, device: Optional[torch.device] = None) -> torch.Tensor:
    """Computes RoPE inverse frequencies for half-split rotary embedding."""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    return inv_freq


def compute_rope_angles(
    positions: Union[int, torch.Tensor],
    dim: int,
    base: float = 10000.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes cos and sin tables for given positions.

    Supports scalar positions or arbitrary-shaped position tensors.
    """
    if isinstance(positions, int):
        positions = torch.tensor([positions], dtype=torch.float32, device=device)
    elif not torch.is_tensor(positions):
        positions = torch.tensor(positions, dtype=torch.float32, device=device)
    else:
        positions = positions.to(dtype=torch.float32)

    inv_freq = compute_inv_freq(dim, base=base, device=device)
    # Outer product between positions and inv_freq
    # positions: (..., 1) or (...)
    angles = positions.unsqueeze(-1) * inv_freq.unsqueeze(0)  # (..., dim // 2)
    angles = torch.cat([angles, angles], dim=-1)  # (..., dim)

    cos = torch.cos(angles).to(dtype=dtype)
    sin = torch.sin(angles).to(dtype=dtype)
    return cos, sin


def apply_rope_half(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: Optional[int] = None,
) -> torch.Tensor:
    """Applies standard RoPE to tensor x using precomputed cos and sin."""
    if unsqueeze_dim is not None:
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


def apply_adjoint_rope(
    q_rot: torch.Tensor,
    delta: Union[int, torch.Tensor],
    dim: Optional[int] = None,
    base: float = 10000.0,
    cos: Optional[torch.Tensor] = None,
    sin: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Applies Query-Side Adjoint RoPE R(-Delta) to an already-rotated query vector.

    Math:
        R(-Delta) q_rot = q_rot * cos(Delta * theta) - rotate_half(q_rot) * sin(Delta * theta)

    This counter-rotates the query by Delta, mathematically equating:
        <R(-Delta) q_rot, k_canonical> == <q_rot, R(Delta) k_canonical>
    """
    head_dim = dim if dim is not None else q_rot.shape[-1]

    if cos is None or sin is None:
        cos, sin = compute_rope_angles(delta, head_dim, base=base, device=q_rot.device, dtype=q_rot.dtype)

    # If cos and sin don't match rank of q_rot, broadcast appropriately
    while cos.dim() < q_rot.dim():
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)

    # R(-Delta) = cos(-theta) + rotate_half * sin(-theta) = cos(theta) - rotate_half * sin(theta)
    return (q_rot * cos) - (rotate_half(q_rot) * sin)


def precompute_freqs_cis(
    dim: int,
    end: int,
    theta: float = 10000.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precomputes cos and sin tables up to max sequence length `end`."""
    positions = torch.arange(end, device=device, dtype=torch.float32)
    return compute_rope_angles(positions, dim, base=theta, device=device, dtype=dtype)


def verify_adjoint_identity(
    q: torch.Tensor,
    k: torch.Tensor,
    n: int,
    p: int,
    delta: int,
    dim: int,
    base: float = 10000.0,
    atol: float = 1e-4,
) -> Tuple[bool, float, float, float]:
    """Mathematically verifies the Adjoint RoPE identity:

        <R(n)q, R(p + delta)k> == <R(-delta) R(n)q, R(p)k>

    Returns:
        (passed, global_score, adjoint_score, max_error)
    """
    cos_n, sin_n = compute_rope_angles(n, dim, base=base, device=q.device, dtype=q.dtype)
    cos_p, sin_p = compute_rope_angles(p, dim, base=base, device=k.device, dtype=k.dtype)
    cos_target, sin_target = compute_rope_angles(p + delta, dim, base=base, device=k.device, dtype=k.dtype)

    # Global standard rotation
    q_global = apply_rope_half(q, cos_n, sin_n)
    k_global = apply_rope_half(k, cos_target, sin_target)
    score_global = torch.sum(q_global * k_global, dim=-1)

    # Canonical k rotated at position p
    k_canon = apply_rope_half(k, cos_p, sin_p)
    # Adjoint query counter-rotated by -delta
    q_adj = apply_adjoint_rope(q_global, delta, dim=dim, base=base)
    score_adj = torch.sum(q_adj * k_canon, dim=-1)

    err = torch.max(torch.abs(score_global - score_adj)).item()
    passed = bool(err <= atol)
    return passed, score_global.mean().item(), score_adj.mean().item(), err
