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
        triton_adjoint_paged_attention_int8,
        triton_adjoint_paged_attention_int4,
    )
    from .triton_k8v4_paged_attn import triton_adjoint_paged_attention_k8v4
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

    if _HAS_TRITON and prefer_triton and q.is_cuda and k_pool.is_cuda and k_scales is not None and v_scales is not None and k_pool.dtype == torch.int8 and v_pool.dtype == torch.uint8:
        try:
            return triton_adjoint_paged_attention_k8v4(q=q, k_pool=k_pool, v_pool=v_pool, k_scales=k_scales, v_scales=v_scales, block_tables=block_tables, offset_tables=offset_tables, seq_lens=seq_lens, sm_scale=sm_scale, base=base, block_start_tables=block_start_tables, valid_lens_tables=valid_lens_tables, delta_bias=delta_bias)
        except Exception:
            pass

    can_use_triton_int8 = (
        _HAS_TRITON
        and prefer_triton
        and q.is_cuda
        and k_pool.is_cuda
        and k_scales is not None
        and v_scales is not None
        and k_pool.dtype == torch.int8
        and v_pool.dtype == torch.int8
    )

    if can_use_triton_int8:
        try:
            return triton_adjoint_paged_attention_int8(
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
        and k_scales is None
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
    can_use_triton_quant = (
        _HAS_TRITON
        and prefer_triton
        and q_bridge.is_cuda
        and k_pool.is_cuda
        and k_scales is not None
        and v_scales is not None
        and (
            (k_pool.dtype == torch.int8 and v_pool.dtype in (torch.int8, torch.uint8))
            or (k_pool.dtype == torch.uint8 and v_pool.dtype == torch.uint8)
        )
    )

    if can_use_triton_quant:
        try:
            bsz, num_heads, num_bridge, head_dim = q_bridge.shape
            if num_bridge == 0:
                return torch.empty_like(q_bridge)
            max_blocks = block_tables.shape[1]
            block_size = k_pool.shape[2]
            bm = bsz * num_bridge

            q_flat = q_bridge.transpose(1, 2).reshape(bm, num_heads, 1, head_dim).contiguous()
            seq_lens_flat = torch.clamp(bridge_positions.reshape(bm) + 1, min=0).to(torch.int32).contiguous()
            bt_flat = block_tables.unsqueeze(1).expand(bsz, num_bridge, max_blocks).reshape(bm, max_blocks).contiguous()
            ot_flat = offset_tables.unsqueeze(1).expand(bsz, num_bridge, max_blocks).reshape(bm, max_blocks).contiguous()

            if block_start_tables is not None:
                bs_flat = block_start_tables.unsqueeze(1).expand(bsz, num_bridge, max_blocks).reshape(bm, max_blocks).contiguous()
            else:
                bs_flat = (
                    torch.arange(max_blocks, device=q_bridge.device, dtype=torch.int32) * block_size
                ).unsqueeze(0).expand(bm, -1).contiguous()

            if valid_lens_tables is not None:
                vl_flat = valid_lens_tables.unsqueeze(1).expand(bsz, num_bridge, max_blocks).reshape(bm, max_blocks).contiguous()
            else:
                vl_flat = torch.full((bm, max_blocks), block_size, device=q_bridge.device, dtype=torch.int32)

            fn = triton_adjoint_paged_attention_int4 if k_pool.dtype == torch.uint8 else (triton_adjoint_paged_attention_k8v4 if v_pool.dtype == torch.uint8 else triton_adjoint_paged_attention_int8)
            out_flat = fn(
                q=q_flat,
                k_pool=k_pool,
                v_pool=v_pool,
                k_scales=k_scales,
                v_scales=v_scales,
                block_tables=bt_flat,
                offset_tables=ot_flat,
                seq_lens=seq_lens_flat,
                sm_scale=sm_scale,
                base=base,
                block_start_tables=bs_flat,
                valid_lens_tables=vl_flat,
                delta_bias=delta_bias,
            )
            out_bridge = out_flat.squeeze(2).reshape(bsz, num_bridge, num_heads, head_dim).transpose(1, 2).contiguous()
            if (bridge_positions < 0).any():
                valid_pos_mask = (bridge_positions >= 0).unsqueeze(1).unsqueeze(-1)
                out_bridge = torch.where(valid_pos_mask, out_bridge, torch.zeros_like(out_bridge))
            return out_bridge
        except Exception:
            pass

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
