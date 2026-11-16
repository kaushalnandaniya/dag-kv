"""Fused Ragged Micro-Prefill Kernel for 8% Bridge Tokens (sm_75 Turing)."""

import math
from typing import Optional
import torch
import triton
import triton.language as tl
from .triton_adjoint_paged_attn import get_cos_sin_cache


@triton.jit
def _triton_sparse_bridge_micro_prefill_kernel(
    Q_bridge_ptr,       # (batch_size, num_heads, num_bridge, head_dim) FP16
    Bridge_pos_ptr,     # (batch_size, num_bridge) INT32
    K_pool_ptr,         # (max_blocks, num_kv_heads, BLOCK_SIZE, head_dim) FP16
    V_pool_ptr,         # (max_blocks, num_kv_heads, BLOCK_SIZE, head_dim) FP16
    Out_bridge_ptr,     # (batch_size, num_heads, num_bridge, head_dim) FP16
    Block_table_ptr,    # (batch_size, max_blocks) INT32
    Offset_table_ptr,   # (batch_size, max_blocks) INT32
    Block_start_ptr,    # (batch_size, max_blocks) INT32
    Valid_lens_ptr,     # (batch_size, max_blocks) INT32
    Cos_cache_ptr,      # (max_delta_span, head_dim) FP16
    Sin_cache_ptr,      # (max_delta_span, head_dim) FP16
    sm_scale: tl.constexpr,
    delta_bias: tl.constexpr,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_bpb, stride_bpm,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ob, stride_oh, stride_om, stride_od,
    stride_btb, stride_btn,
    stride_otb, stride_otn,
    stride_bsb, stride_bsn,
    stride_vlb, stride_vln,
    stride_cos_d, stride_cos_dim,
    stride_sin_d, stride_sin_dim,
    num_total_blocks: int,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    cur_batch = tl.program_id(0)
    cur_head = tl.program_id(1)
    bridge_idx = tl.program_id(2)

    kv_head = cur_head // (num_heads // num_kv_heads)

    global_pos = tl.load(Bridge_pos_ptr + cur_batch * stride_bpb + bridge_idx * stride_bpm)
    if global_pos < 0:
        return

    offs_d = tl.arange(0, HEAD_DIM)
    rot_idx = (offs_d + HEAD_DIM // 2) % HEAD_DIM
    rot_sign = tl.where(offs_d < HEAD_DIM // 2, -1.0, 1.0).to(tl.float32)

    q_offset = cur_batch * stride_qb + cur_head * stride_qh + bridge_idx * stride_qm + offs_d * stride_qd
    q_rot_offset = cur_batch * stride_qb + cur_head * stride_qh + bridge_idx * stride_qm + rot_idx * stride_qd
    q = tl.load(Q_bridge_ptr + q_offset).to(tl.float32)
    q_rot = (tl.load(Q_bridge_ptr + q_rot_offset).to(tl.float32)) * rot_sign

    m_prev = -float("inf")
    l_prev = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    causal_limit = global_pos + 1
    offs_n = tl.arange(0, BLOCK_SIZE)

    for b in range(num_total_blocks):
        blk_start = tl.load(Block_start_ptr + cur_batch * stride_bsb + b * stride_bsn)
        valid_len = tl.load(Valid_lens_ptr + cur_batch * stride_vlb + b * stride_vln)
        if blk_start < causal_limit and valid_len > 0:
            phys_block = tl.load(Block_table_ptr + cur_batch * stride_btb + b * stride_btn)
            delta = tl.load(Offset_table_ptr + cur_batch * stride_otb + b * stride_otn)

            delta_row = delta + delta_bias
            cos_val = tl.load(Cos_cache_ptr + delta_row * stride_cos_d + offs_d * stride_cos_dim).to(tl.float32)
            sin_val = tl.load(Sin_cache_ptr + delta_row * stride_sin_d + offs_d * stride_sin_dim).to(tl.float32)

            q_adj = q * cos_val - q_rot * sin_val

            token_indices = blk_start + offs_n
            valid_mask = (offs_n < valid_len) & (token_indices < causal_limit)
            k_mask = valid_mask[:, None]

            k_ptrs = (
                K_pool_ptr
                + phys_block * stride_kb
                + kv_head * stride_kh
                + offs_n[:, None] * stride_kn
                + offs_d[None, :] * stride_kd
            )
            k = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)

            scores = tl.sum(k * q_adj[None, :], 1) * sm_scale
            scores = tl.where(valid_mask, scores, -float("inf"))

            m_curr = tl.max(scores, 0)
            m_new = tl.maximum(m_prev, m_curr)
            p = tl.exp(scores - m_new)
            p = tl.where(valid_mask, p, 0.0)

            alpha = tl.exp(m_prev - m_new)
            l_new = alpha * l_prev + tl.sum(p, 0)

            v_ptrs = (
                V_pool_ptr
                + phys_block * stride_vb
                + kv_head * stride_vh
                + offs_n[:, None] * stride_vn
                + offs_d[None, :] * stride_vd
            )
            v = tl.load(v_ptrs, mask=k_mask, other=0.0).to(tl.float32)
            v_out = tl.sum(p[:, None] * v, 0)

            acc = acc * alpha + v_out
            m_prev = m_new
            l_prev = l_new

    l_safe = tl.where(l_prev > 0.0, l_prev, 1.0)
    out = (acc / l_safe).to(tl.float16)

    out_offset = (
        cur_batch * stride_ob
        + cur_head * stride_oh
        + bridge_idx * stride_om
        + offs_d * stride_od
    )
    tl.store(Out_bridge_ptr + out_offset, out)


def triton_sparse_bridge_attention(
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
) -> torch.Tensor:
    assert q_bridge.dtype == torch.float16, "sm_75 constraint: q_bridge must be float16"
    assert k_pool.dtype == torch.float16, "sm_75 constraint: k_pool must be float16"
    assert v_pool.dtype == torch.float16, "sm_75 constraint: v_pool must be float16"

    batch_size, num_heads, num_bridge, head_dim = q_bridge.shape
    num_kv_heads = k_pool.shape[1]
    block_size = k_pool.shape[2]
    max_blocks = block_tables.shape[1]

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    if block_start_tables is None:
        block_start_tables = (
            torch.arange(max_blocks, device=q_bridge.device, dtype=torch.int32) * block_size
        ).unsqueeze(0).expand(batch_size, -1).contiguous()
    if valid_lens_tables is None:
        valid_lens_tables = torch.full(
            (batch_size, max_blocks), block_size, device=q_bridge.device, dtype=torch.int32
        )

    cos_cache, sin_cache, delta_bias = get_cos_sin_cache(
        head_dim=head_dim, base=base, max_delta=16384, device=q_bridge.device
    )

    out_bridge = torch.empty_like(q_bridge)

    grid = (batch_size, num_heads, num_bridge)
    with torch.cuda.device(q_bridge.device):
        _triton_sparse_bridge_micro_prefill_kernel[grid](
            q_bridge, bridge_positions,
            k_pool, v_pool, out_bridge,
            block_tables, offset_tables, block_start_tables, valid_lens_tables,
            cos_cache, sin_cache,
            sm_scale=sm_scale,
            delta_bias=delta_bias,
            stride_qb=q_bridge.stride(0), stride_qh=q_bridge.stride(1), stride_qm=q_bridge.stride(2), stride_qd=q_bridge.stride(3),
            stride_bpb=bridge_positions.stride(0), stride_bpm=bridge_positions.stride(1),
            stride_kb=k_pool.stride(0), stride_kh=k_pool.stride(1), stride_kn=k_pool.stride(2), stride_kd=k_pool.stride(3),
            stride_vb=v_pool.stride(0), stride_vh=v_pool.stride(1), stride_vn=v_pool.stride(2), stride_vd=v_pool.stride(3),
            stride_ob=out_bridge.stride(0), stride_oh=out_bridge.stride(1), stride_om=out_bridge.stride(2), stride_od=out_bridge.stride(3),
            stride_btb=block_tables.stride(0), stride_btn=block_tables.stride(1),
            stride_otb=offset_tables.stride(0), stride_otn=offset_tables.stride(1),
            stride_bsb=block_start_tables.stride(0), stride_bsn=block_start_tables.stride(1),
            stride_vlb=valid_lens_tables.stride(0), stride_vln=valid_lens_tables.stride(1),
            stride_cos_d=cos_cache.stride(0), stride_cos_dim=cos_cache.stride(1),
            stride_sin_d=sin_cache.stride(0), stride_sin_dim=sin_cache.stride(1),
            num_total_blocks=max_blocks,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            BLOCK_SIZE=block_size,
            HEAD_DIM=head_dim,
            num_warps=4,
            num_stages=2,
        )

    return out_bridge


def torch_sparse_bridge_reference(
    q_bridge: torch.Tensor,
    bridge_positions: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    block_tables: torch.Tensor,
    offset_tables: torch.Tensor,
    sm_scale: Optional[float] = None,
    base: float = 10000.0,
) -> torch.Tensor:
    from dag_kv.core.rope_adjoint import apply_adjoint_rope

    batch_size, num_heads, num_bridge, head_dim = q_bridge.shape
    num_kv_heads = k_pool.shape[1]
    block_size = k_pool.shape[2]
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    out = torch.zeros_like(q_bridge)
    gqa_ratio = num_heads // num_kv_heads

    for b in range(batch_size):
        for m in range(num_bridge):
            gpos = bridge_positions[b, m].item()
            if gpos < 0:
                continue
            causal_limit = gpos + 1
            num_blocks = (causal_limit + block_size - 1) // block_size

            for h in range(num_heads):
                kv_h = h // gqa_ratio
                q_vec = q_bridge[b, h, m]

                all_scores = []
                all_v = []

                for blk_idx in range(num_blocks):
                    phys_id = block_tables[b, blk_idx].item()
                    delta = offset_tables[b, blk_idx].item()
                    valid_tokens = min(block_size, causal_limit - blk_idx * block_size)

                    q_adj = apply_adjoint_rope(q_vec, delta, dim=head_dim, base=base).view(1, head_dim)
                    k_blk = k_pool[phys_id, kv_h, :valid_tokens]
                    v_blk = v_pool[phys_id, kv_h, :valid_tokens]

                    score_blk = (q_adj * k_blk).sum(dim=-1).view(-1) * sm_scale
                    all_scores.append(score_blk)
                    all_v.append(v_blk)

                scores_cat = torch.cat(all_scores, dim=0)
                v_cat = torch.cat(all_v, dim=0)

                probs = torch.softmax(scores_cat.to(torch.float32), dim=-1).to(torch.float16)
                out_vec = (probs.unsqueeze(-1) * v_cat).sum(dim=0)
                out[b, h, m] = out_vec

    return out
