"""Universal Kernel Dispatcher for DAG-KV.

Dynamically selects between hardware-optimized Triton JIT kernels (NVIDIA/AMD)
and pure PyTorch native fallbacks (Apple Silicon MPS, CPU, and Quantized pools).
"""

from typing import Optional
import torch

from .pytorch_fallbacks import pytorch_adjoint_paged_attention, pytorch_sparse_bridge_attention

_HAS_TRITON = False
try:
    import triton
    import triton.language as tl
    from .triton_adjoint_paged_attn import (
        triton_adjoint_paged_attention,
        triton_adjoint_paged_attention_int4,
    )
    from .triton_sparse_stitch import triton_sparse_bridge_attention
    _HAS_TRITON = True
except (ImportError, Exception):
    _HAS_TRITON = False


def is_triton_available() -> bool:
    """Returns True if Triton runtime is available and importable."""
    return _HAS_TRITON


def adjoint_paged_attention(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    block_tables: torch.Tensor,
    offset_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    sm_scale: Optional[float] = None,
    base: float = 10000.0,
    block_start_tables: Optional[torch.Tensor] = None,
    valid_lens_tables: Optional[torch.Tensor] = None,
    k_scales: Optional[torch.Tensor] = None,
    v_scales: Optional[torch.Tensor] = None,
    prefer_triton: bool = True,
    delta_bias: int = 32768,
) -> torch.Tensor:
    """Universal Adjoint PagedAttention with automated backend dispatch.

    Dispatches to Triton on CUDA devices for maximum throughput.
    Falls back to vectorized PyTorch on CPU, Apple Silicon (MPS), or INT8 pools.
    """
    can_use_triton_int4 = (
        _HAS_TRITON
        and prefer_triton
        and q.is_cuda
        and k_pool.is_cuda
        and k_scales is not None
        and v_scales is not None
        and k_pool.dtype == torch.uint8
        and v_pool.dtype == torch.uint8
    )

    if can_use_triton_int4:
        try:
            return triton_adjoint_paged_attention_int4(
                q=q,
                k_pool=k_pool,
                v_pool=v_pool,
                k_scales=k_scales,
                v_scales=v_scales,
                block_tables=block_tables,
                offset_tables=offset_tables,
                seq_lens=seq_lens,
                sm_scale=sm_scale,
                base=base,
                block_start_tables=block_start_tables,
                valid_lens_tables=valid_lens_tables,
                delta_bias=delta_bias,
            )
        except Exception:
            pass

    can_use_triton = (
        _HAS_TRITON
        and prefer_triton
        and q.is_cuda
        and k_pool.is_cuda
        and k_scales is None  # INT8 scales use PyTorch dequant fallback
        and k_pool.dtype in (torch.float16, torch.bfloat16)
    )

    if can_use_triton:
        try:
            return triton_adjoint_paged_attention(
                q=q,
                k_pool=k_pool,
                v_pool=v_pool,
                block_tables=block_tables,
                offset_tables=offset_tables,
                seq_lens=seq_lens,
                sm_scale=sm_scale,
                base=base,
                block_start_tables=block_start_tables,
                valid_lens_tables=valid_lens_tables,
                delta_bias=delta_bias,
            )
        except Exception:
            # Fall back safely if Triton compilation fails on unsupported device
            pass

    return pytorch_adjoint_paged_attention(
        q=q,
        k_pool=k_pool,
        v_pool=v_pool,
        block_tables=block_tables,
        offset_tables=offset_tables,
        seq_lens=seq_lens,
        sm_scale=sm_scale,
        base=base,
        block_start_tables=block_start_tables,
        valid_lens_tables=valid_lens_tables,
        k_scales=k_scales,
        v_scales=v_scales,
        delta_bias=delta_bias,
    )


def sparse_bridge_attention(
    q_bridge: torch.Tensor,
    bridge_positions: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    block_tables: torch.Tensor,
    offset_tables: torch.Tensor,
    sm_scale: Optional[float] = None,
    base: float = 10000.0,
    block_start_tables: Optional[torch.Tensor] = None,
    valid_lens_tables: Optional[torch.Tensor] = None,
    k_scales: Optional[torch.Tensor] = None,
    v_scales: Optional[torch.Tensor] = None,
    prefer_triton: bool = True,
    delta_bias: int = 32768,
) -> torch.Tensor:
    """Universal Sparse Bridge Ragged Micro-Prefill with automated backend dispatch."""
    can_use_triton = (
        _HAS_TRITON
        and prefer_triton
        and q_bridge.is_cuda
        and k_pool.is_cuda
        and k_scales is None
        and k_pool.dtype in (torch.float16, torch.bfloat16)
    )

    if can_use_triton:
        try:
            return triton_sparse_bridge_attention(
                q_bridge=q_bridge,
                bridge_positions=bridge_positions,
                k_pool=k_pool,
                v_pool=v_pool,
                block_tables=block_tables,
                offset_tables=offset_tables,
                sm_scale=sm_scale,
                base=base,
                block_start_tables=block_start_tables,
                valid_lens_tables=valid_lens_tables,
                delta_bias=delta_bias,
            )
        except Exception:
            pass

    return pytorch_sparse_bridge_attention(
        q_bridge=q_bridge,
        bridge_positions=bridge_positions,
        k_pool=k_pool,
        v_pool=v_pool,
        block_tables=block_tables,
        offset_tables=offset_tables,
        sm_scale=sm_scale,
        base=base,
        block_start_tables=block_start_tables,
        valid_lens_tables=valid_lens_tables,
        k_scales=k_scales,
        v_scales=v_scales,
        delta_bias=delta_bias,
    )
