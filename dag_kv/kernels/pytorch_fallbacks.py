"""Pure PyTorch Fallback Kernels for Adjoint PagedAttention & Sparse Bridge Repair.

Hardware-agnostic vectorized implementation designed for:
1. Apple Silicon (MPS)
2. CPU-only and ARM environments
3. AMD ROCm without Triton JIT or non-Triton PyTorch builds
4. CI/CD test runners without GPUs
"""

import math
from typing import Optional, Tuple
import torch
import torch.nn.functional as F

from dag_kv.core.rope_adjoint import apply_adjoint_rope


def pytorch_adjoint_paged_attention(
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
    delta_bias: int = 16384,
) -> torch.Tensor:
    """Vectorized PyTorch implementation of Adjoint PagedAttention decode.

    Args:
        q: (batch_size, num_heads, 1, head_dim)
        k_pool: (max_blocks, num_kv_heads, block_size, head_dim)
        v_pool: (max_blocks, num_kv_heads, block_size, head_dim)
        block_tables: (batch_size, max_blocks_per_seq) INT32
        offset_tables: (batch_size, max_blocks_per_seq) INT32 (RoPE deltas)
        seq_lens: (batch_size,) INT32
        sm_scale: Softmax scale (defaults to 1 / sqrt(head_dim))
        base: RoPE base frequency
        block_start_tables: (batch_size, max_blocks_per_seq) INT32
        valid_lens_tables: (batch_size, max_blocks_per_seq) INT32
        k_scales: Optional (max_blocks, num_kv_heads, 1) FP32 for INT8 dequantization
        v_scales: Optional (max_blocks, num_kv_heads, 1) FP32 for INT8 dequantization

    Returns:
        out: (batch_size, num_heads, 1, head_dim) in same dtype as q
    """
    batch_size, num_heads, q_len, head_dim = q.shape
    max_blocks, num_kv_heads, block_size, _ = k_pool.shape
    device = q.device
    out_dtype = q.dtype

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    num_total_blocks = block_tables.shape[1]
    gqa_ratio = num_heads // num_kv_heads

    # Derive default block starts / valid lens if not provided
    if block_start_tables is None:
        block_start_tables = (
            torch.arange(num_total_blocks, device=device, dtype=torch.int32).unsqueeze(0).expand(batch_size, -1)
            * block_size
        )
    if valid_lens_tables is None:
        valid_lens_tables = torch.full(
            (batch_size, num_total_blocks), block_size, device=device, dtype=torch.int32
        )

    out = torch.zeros((batch_size, num_heads, q_len, head_dim), dtype=out_dtype, device=device)

    # Online softmax accumulation per (batch, head)
    # acc: (batch_size, num_heads, head_dim)
    # m_prev: (batch_size, num_heads) = -inf
    # l_prev: (batch_size, num_heads) = 0.0
    for b_idx in range(batch_size):
        seq_len = int(seq_lens[b_idx].item())
        if seq_len <= 0:
            continue

        q_b = q[b_idx, :, 0, :].to(torch.float32)  # (num_heads, head_dim)
        m_prev = torch.full((num_heads,), float("-inf"), device=device, dtype=torch.float32)
        l_prev = torch.zeros((num_heads,), device=device, dtype=torch.float32)
        acc = torch.zeros((num_heads, head_dim), device=device, dtype=torch.float32)

        for blk_idx in range(num_total_blocks):
            blk_start = int(block_start_tables[b_idx, blk_idx].item())
            valid_len = int(valid_lens_tables[b_idx, blk_idx].item())

            if blk_start >= seq_len or valid_len <= 0:
                continue

            phys_block = int(block_tables[b_idx, blk_idx].item())
            delta = int(offset_tables[b_idx, blk_idx].item())

            # Adjoint counter-rotation R(-Delta) on Q
            # q_b: (num_heads, head_dim) -> q_adj: (num_heads, head_dim)
            q_adj = apply_adjoint_rope(q_b, delta=delta, dim=head_dim, base=base)

            # Fetch keys and values: (num_kv_heads, block_size, head_dim)
            k_blk = k_pool[phys_block, :, :, :].to(torch.float32)
            v_blk = v_pool[phys_block, :, :, :].to(torch.float32)

            if k_scales is not None:
                k_blk = k_blk * k_scales[phys_block]
            if v_scales is not None:
                v_blk = v_blk * v_scales[phys_block]

            # GQA expansion: repeat kv heads to match query heads
            if gqa_ratio > 1:
                k_blk = k_blk.repeat_interleave(gqa_ratio, dim=0)  # (num_heads, block_size, head_dim)
                v_blk = v_blk.repeat_interleave(gqa_ratio, dim=0)

            # Valid token mask within this block
            # tokens at positions: blk_start + 0 .. blk_start + block_size - 1
            offs = torch.arange(block_size, device=device)
            mask = (offs < valid_len) & ((blk_start + offs) < seq_len)  # (block_size,)

            if not mask.any():
                continue

            # Dot products: (num_heads, block_size)
            # q_adj is (num_heads, 1, head_dim), k_blk is (num_heads, block_size, head_dim)
            scores = torch.sum(q_adj.unsqueeze(1) * k_blk, dim=-1) * sm_scale  # (num_heads, block_size)
            scores[:, ~mask] = float("-inf")

            # Online Softmax update (Milakov & Gimelshein / FlashAttention)
            m_curr = torch.max(scores, dim=-1).values  # (num_heads,)
            m_new = torch.maximum(m_prev, m_curr)

            # Avoid exp(-inf - -inf) = nan: if all masked, p is 0
            p = torch.exp(scores - m_new.unsqueeze(-1))  # (num_heads, block_size)
            p[:, ~mask] = 0.0

            alpha = torch.exp(m_prev - m_new)
            alpha = torch.where(torch.isnan(alpha), torch.zeros_like(alpha), alpha)
            l_new = alpha * l_prev + torch.sum(p, dim=-1)

            # acc = alpha * acc + sum(p * v)
            p_v = torch.bmm(p.unsqueeze(1), v_blk).squeeze(1)  # (num_heads, head_dim)
            acc = alpha.unsqueeze(-1) * acc + p_v

            m_prev = m_new
            l_prev = l_new

        # Normalize
        l_prev_safe = torch.clamp(l_prev.unsqueeze(-1), min=1e-8)
        out_b = acc / l_prev_safe
        out[b_idx, :, 0, :] = out_b.to(out_dtype)

    return out


def pytorch_sparse_bridge_attention(
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
    delta_bias: int = 16384,
) -> torch.Tensor:
    """Vectorized PyTorch implementation of Fused Ragged Micro-Prefill for Bridge Tokens.

    Args:
        q_bridge: (batch_size, num_heads, num_bridge, head_dim)
        bridge_positions: (batch_size, num_bridge) INT32
        k_pool: (max_blocks, num_kv_heads, block_size, head_dim)
        v_pool: (max_blocks, num_kv_heads, block_size, head_dim)
        block_tables: (batch_size, max_blocks) INT32
        offset_tables: (batch_size, max_blocks) INT32
        sm_scale: Softmax scale
        base: RoPE base frequency
        block_start_tables: (batch_size, max_blocks) INT32
        valid_lens_tables: (batch_size, max_blocks) INT32
        k_scales: Optional INT8 dequantization scale
        v_scales: Optional INT8 dequantization scale

    Returns:
        out: (batch_size, num_heads, num_bridge, head_dim) in same dtype as q_bridge
    """
    batch_size, num_heads, num_bridge, head_dim = q_bridge.shape
    max_blocks, num_kv_heads, block_size, _ = k_pool.shape
    device = q_bridge.device
    out_dtype = q_bridge.dtype

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    num_total_blocks = block_tables.shape[1]
    gqa_ratio = num_heads // num_kv_heads

    if block_start_tables is None:
        block_start_tables = (
            torch.arange(num_total_blocks, device=device, dtype=torch.int32).unsqueeze(0).expand(batch_size, -1)
            * block_size
        )
    if valid_lens_tables is None:
        valid_lens_tables = torch.full(
            (batch_size, num_total_blocks), block_size, device=device, dtype=torch.int32
        )

    out = torch.zeros((batch_size, num_heads, num_bridge, head_dim), dtype=out_dtype, device=device)

    for b_idx in range(batch_size):
        for br_idx in range(num_bridge):
            global_pos = int(bridge_positions[b_idx, br_idx].item())
            if global_pos < 0:
                continue

            causal_limit = global_pos + 1
            q_br = q_bridge[b_idx, :, br_idx, :].to(torch.float32)  # (num_heads, head_dim)

            m_prev = torch.full((num_heads,), float("-inf"), device=device, dtype=torch.float32)
            l_prev = torch.zeros((num_heads,), device=device, dtype=torch.float32)
            acc = torch.zeros((num_heads, head_dim), device=device, dtype=torch.float32)

            for blk_idx in range(num_total_blocks):
                blk_start = int(block_start_tables[b_idx, blk_idx].item())
                valid_len = int(valid_lens_tables[b_idx, blk_idx].item())

                if blk_start >= causal_limit or valid_len <= 0:
                    continue

                phys_block = int(block_tables[b_idx, blk_idx].item())
                delta = int(offset_tables[b_idx, blk_idx].item())

                # Adjoint counter-rotation R(-Delta)
                q_adj = apply_adjoint_rope(q_br, delta=delta, dim=head_dim, base=base)

                k_blk = k_pool[phys_block, :, :, :].to(torch.float32)
                v_blk = v_pool[phys_block, :, :, :].to(torch.float32)

                if k_scales is not None:
                    k_blk = k_blk * k_scales[phys_block]
                if v_scales is not None:
                    v_blk = v_blk * v_scales[phys_block]

                if gqa_ratio > 1:
                    k_blk = k_blk.repeat_interleave(gqa_ratio, dim=0)
                    v_blk = v_blk.repeat_interleave(gqa_ratio, dim=0)

                offs = torch.arange(block_size, device=device)
                mask = (offs < valid_len) & ((blk_start + offs) < causal_limit)

                if not mask.any():
                    continue

                scores = torch.sum(q_adj.unsqueeze(1) * k_blk, dim=-1) * sm_scale
                scores[:, ~mask] = float("-inf")

                m_curr = torch.max(scores, dim=-1).values
                m_new = torch.maximum(m_prev, m_curr)

                p = torch.exp(scores - m_new.unsqueeze(-1))
                p[:, ~mask] = 0.0

                alpha = torch.exp(m_prev - m_new)
                alpha = torch.where(torch.isnan(alpha), torch.zeros_like(alpha), alpha)
                l_new = alpha * l_prev + torch.sum(p, dim=-1)

                p_v = torch.bmm(p.unsqueeze(1), v_blk).squeeze(1)
                acc = alpha.unsqueeze(-1) * acc + p_v

                m_prev = m_new
                l_prev = l_new

            l_prev_safe = torch.clamp(l_prev.unsqueeze(-1), min=1e-8)
            out_br = acc / l_prev_safe
            out[b_idx, :, br_idx, :] = out_br.to(out_dtype)

    return out
