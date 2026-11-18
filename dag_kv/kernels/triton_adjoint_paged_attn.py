"""Triton PagedAttention with Query-Side Adjoint RoPE for sm_75 (Turing Architecture)."""

import math
from typing import Optional, Tuple
import torch
import triton
import triton.language as tl


@triton.jit
def _triton_adjoint_paged_attn_decode_kernel(
    Q_ptr,              # (batch_size, num_heads, 1, head_dim) FP16
    K_pool_ptr,         # (max_blocks, num_kv_heads, BLOCK_SIZE, head_dim) FP16
    V_pool_ptr,         # (max_blocks, num_kv_heads, BLOCK_SIZE, head_dim) FP16
    Out_ptr,            # (batch_size, num_heads, 1, head_dim) FP16
    Block_table_ptr,    # (batch_size, max_blocks_per_seq) INT32
    Offset_table_ptr,   # (batch_size, max_blocks_per_seq) INT32
    Block_start_ptr,    # (batch_size, max_blocks_per_seq) INT32
    Valid_lens_ptr,     # (batch_size, max_blocks_per_seq) INT32
    Seq_lens_ptr,       # (batch_size,) INT32
    Cos_cache_ptr,      # (max_delta_span, head_dim) FP16
    Sin_cache_ptr,      # (max_delta_span, head_dim) FP16
    sm_scale: tl.constexpr,
    delta_bias: tl.constexpr,
    stride_qb, stride_qh, stride_qm, stride_qd,
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
    kv_head = cur_head // (num_heads // num_kv_heads)

    seq_len = tl.load(Seq_lens_ptr + cur_batch)
    if seq_len <= 0:
        return

    offs_d = tl.arange(0, HEAD_DIM)
    rot_idx = (offs_d + HEAD_DIM // 2) % HEAD_DIM
    rot_sign = tl.where(offs_d < HEAD_DIM // 2, -1.0, 1.0).to(tl.float32)

    q_offset = cur_batch * stride_qb + cur_head * stride_qh + offs_d * stride_qd
    q_rot_offset = cur_batch * stride_qb + cur_head * stride_qh + rot_idx * stride_qd
    q = tl.load(Q_ptr + q_offset).to(tl.float32)
    q_rot = (tl.load(Q_ptr + q_rot_offset).to(tl.float32)) * rot_sign

    m_prev = -float("inf")
    l_prev = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    offs_n = tl.arange(0, BLOCK_SIZE)

    for b in range(num_total_blocks):
        blk_start = tl.load(Block_start_ptr + cur_batch * stride_bsb + b * stride_bsn)
        valid_len = tl.load(Valid_lens_ptr + cur_batch * stride_vlb + b * stride_vln)
        if blk_start < seq_len and valid_len > 0:
            phys_block = tl.load(Block_table_ptr + cur_batch * stride_btb + b * stride_btn)
            delta = tl.load(Offset_table_ptr + cur_batch * stride_otb + b * stride_otn)

            delta_row = delta + delta_bias
            cos_val = tl.load(Cos_cache_ptr + delta_row * stride_cos_d + offs_d * stride_cos_dim).to(tl.float32)
            sin_val = tl.load(Sin_cache_ptr + delta_row * stride_sin_d + offs_d * stride_sin_dim).to(tl.float32)

            q_adj = q * cos_val - q_rot * sin_val

            token_indices = blk_start + offs_n
            valid_mask = (offs_n < valid_len) & (token_indices < seq_len)
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

    out_offset = cur_batch * stride_ob + cur_head * stride_oh + offs_d * stride_od
    tl.store(Out_ptr + out_offset, out)


_GLOBAL_COS_SIN_CACHE = {}


def get_cos_sin_cache(
    head_dim: int,
    base: float = 10000.0,
    max_delta: int = 16384,
    device: torch.device = torch.device("cuda:0"),
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    key = (head_dim, base, max_delta, device)
    if key in _GLOBAL_COS_SIN_CACHE:
        return _GLOBAL_COS_SIN_CACHE[key]

    delta_bias = max_delta
    deltas = torch.arange(-max_delta, max_delta + 1, device=device, dtype=torch.float32)

    from dag_kv.core.rope_adjoint import compute_rope_angles
    cos, sin = compute_rope_angles(deltas, head_dim, base=base, device=device, dtype=torch.float16)

    _GLOBAL_COS_SIN_CACHE[key] = (cos, sin, delta_bias)
    return cos, sin, delta_bias


def triton_adjoint_paged_attention(
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
) -> torch.Tensor:
    assert q.dtype == torch.float16, "Turing sm_75 constraint: q must be float16"
    assert k_pool.dtype == torch.float16, "Turing sm_75 constraint: k_pool must be float16"
    assert v_pool.dtype == torch.float16, "Turing sm_75 constraint: v_pool must be float16"

    batch_size, num_heads, q_len, head_dim = q.shape
    assert q_len == 1, "Decode PagedAttention expects q_len == 1"
    num_kv_heads = k_pool.shape[1]
    block_size = k_pool.shape[2]
    max_blocks = block_tables.shape[1]

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    if block_start_tables is None:
        block_start_tables = (
            torch.arange(max_blocks, device=q.device, dtype=torch.int32) * block_size
        ).unsqueeze(0).expand(batch_size, -1).contiguous()
    if valid_lens_tables is None:
        valid_lens_tables = torch.clamp(
            seq_lens.unsqueeze(1) - block_start_tables, min=0, max=block_size
        ).to(torch.int32).contiguous()

    cos_cache, sin_cache, delta_bias = get_cos_sin_cache(
        head_dim=head_dim, base=base, max_delta=16384, device=q.device
    )

    out = torch.empty_like(q)

    grid = (batch_size, num_heads)
    with torch.cuda.device(q.device):
        _triton_adjoint_paged_attn_decode_kernel[grid](
            q, k_pool, v_pool, out,
            block_tables, offset_tables, block_start_tables, valid_lens_tables, seq_lens,
            cos_cache, sin_cache,
            sm_scale=sm_scale,
            delta_bias=delta_bias,
            stride_qb=q.stride(0), stride_qh=q.stride(1), stride_qm=q.stride(2), stride_qd=q.stride(3),
            stride_kb=k_pool.stride(0), stride_kh=k_pool.stride(1), stride_kn=k_pool.stride(2), stride_kd=k_pool.stride(3),
            stride_vb=v_pool.stride(0), stride_vh=v_pool.stride(1), stride_vn=v_pool.stride(2), stride_vd=v_pool.stride(3),
            stride_ob=out.stride(0), stride_oh=out.stride(1), stride_om=out.stride(2), stride_od=out.stride(3),
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

    return out



def torch_adjoint_paged_attention_reference(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    block_tables: torch.Tensor,
    offset_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    sm_scale: Optional[float] = None,
    base: float = 10000.0,
) -> torch.Tensor:
    from dag_kv.core.rope_adjoint import apply_adjoint_rope

    batch_size, num_heads, q_len, head_dim = q.shape
    num_kv_heads = k_pool.shape[1]
    block_size = k_pool.shape[2]
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    out = torch.zeros_like(q)
    gqa_ratio = num_heads // num_kv_heads

    for b in range(batch_size):
        slen = seq_lens[b].item()
        if slen <= 0:
            continue
        num_blocks = (slen + block_size - 1) // block_size

        for h in range(num_heads):
            kv_h = h // gqa_ratio
            q_vec = q[b, h, 0]

            all_scores = []
            all_v = []

            for blk_idx in range(num_blocks):
                phys_id = block_tables[b, blk_idx].item()
                delta = offset_tables[b, blk_idx].item()
                valid_tokens = min(block_size, slen - blk_idx * block_size)

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
            out[b, h, 0] = out_vec

    return out



@triton.jit
def _triton_adjoint_paged_attn_int4_decode_kernel(
    Q_ptr,              # (batch_size, num_heads, 1, head_dim) FP16
    K_pool_ptr,         # (max_blocks, num_kv_heads, BLOCK_SIZE, head_dim // 2) UINT8
    V_pool_ptr,         # (max_blocks, num_kv_heads, BLOCK_SIZE, head_dim // 2) UINT8
    K_scale_ptr,        # (max_blocks, num_kv_heads, BLOCK_SIZE) FP16
    V_scale_ptr,        # (max_blocks, num_kv_heads, BLOCK_SIZE) FP16
    Out_ptr,            # (batch_size, num_heads, 1, head_dim) FP16
    Block_table_ptr,    # (batch_size, max_blocks_per_seq) INT32
    Offset_table_ptr,   # (batch_size, max_blocks_per_seq) INT32
    Block_start_ptr,    # (batch_size, max_blocks_per_seq) INT32
    Valid_lens_ptr,     # (batch_size, max_blocks_per_seq) INT32
    Seq_lens_ptr,       # (batch_size,) INT32
    Cos_cache_ptr,      # (max_delta_span, head_dim) FP16
    Sin_cache_ptr,      # (max_delta_span, head_dim) FP16
    sm_scale: tl.constexpr,
    delta_bias: tl.constexpr,
    stride_qb, stride_qh, stride_qm, stride_qd,
    stride_kb, stride_kh, stride_kn, stride_kd,
    stride_vb, stride_vh, stride_vn, stride_vd,
    stride_ksb, stride_ksh, stride_ksn,
    stride_vsb, stride_vsh, stride_vsn,
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
    kv_head = cur_head // (num_heads // num_kv_heads)

    seq_len = tl.load(Seq_lens_ptr + cur_batch)
    if seq_len <= 0:
        return

    HALF_DIM: tl.constexpr = HEAD_DIM // 2
    offs_half = tl.arange(0, HALF_DIM)
    offs_even = offs_half * 2
    offs_odd = offs_half * 2 + 1

    rot_idx_even = (offs_even + HALF_DIM) % HEAD_DIM
    rot_sign_even = tl.where(offs_even < HALF_DIM, -1.0, 1.0).to(tl.float32)

    rot_idx_odd = (offs_odd + HALF_DIM) % HEAD_DIM
    rot_sign_odd = tl.where(offs_odd < HALF_DIM, -1.0, 1.0).to(tl.float32)

    q_even = tl.load(Q_ptr + cur_batch * stride_qb + cur_head * stride_qh + offs_even * stride_qd).to(tl.float32)
    q_rot_even = (tl.load(Q_ptr + cur_batch * stride_qb + cur_head * stride_qh + rot_idx_even * stride_qd).to(tl.float32)) * rot_sign_even

    q_odd = tl.load(Q_ptr + cur_batch * stride_qb + cur_head * stride_qh + offs_odd * stride_qd).to(tl.float32)
    q_rot_odd = (tl.load(Q_ptr + cur_batch * stride_qb + cur_head * stride_qh + rot_idx_odd * stride_qd).to(tl.float32)) * rot_sign_odd

    m_prev = -float("inf")
    l_prev = 0.0
    acc_even = tl.zeros([HALF_DIM], dtype=tl.float32)
    acc_odd = tl.zeros([HALF_DIM], dtype=tl.float32)
    offs_n = tl.arange(0, BLOCK_SIZE)

    for b in range(num_total_blocks):
        blk_start = tl.load(Block_start_ptr + cur_batch * stride_bsb + b * stride_bsn)
        valid_len = tl.load(Valid_lens_ptr + cur_batch * stride_vlb + b * stride_vln)
        if blk_start < seq_len and valid_len > 0:
            phys_block = tl.load(Block_table_ptr + cur_batch * stride_btb + b * stride_btn)
            delta = tl.load(Offset_table_ptr + cur_batch * stride_otb + b * stride_otn)

            delta_row = delta + delta_bias
            cos_even = tl.load(Cos_cache_ptr + delta_row * stride_cos_d + offs_even * stride_cos_dim).to(tl.float32)
            sin_even = tl.load(Sin_cache_ptr + delta_row * stride_sin_d + offs_even * stride_sin_dim).to(tl.float32)
            cos_odd = tl.load(Cos_cache_ptr + delta_row * stride_cos_d + offs_odd * stride_cos_dim).to(tl.float32)
            sin_odd = tl.load(Sin_cache_ptr + delta_row * stride_sin_d + offs_odd * stride_sin_dim).to(tl.float32)

            q_adj_even = q_even * cos_even - q_rot_even * sin_even
            q_adj_odd = q_odd * cos_odd - q_rot_odd * sin_odd

            token_indices = blk_start + offs_n
            valid_mask = (offs_n < valid_len) & (token_indices < seq_len)
            k_mask = valid_mask[:, None]

            # Load packed INT4 K vectors (HALF_DIM bytes per token)
            k_ptrs = (
                K_pool_ptr
                + phys_block * stride_kb
                + kv_head * stride_kh
                + offs_n[:, None] * stride_kn
                + offs_half[None, :] * stride_kd
            )
            k_packed = tl.load(k_ptrs, mask=k_mask, other=0)

            # Load K scales: (BLOCK_SIZE,)
            k_scale_ptrs = (
                K_scale_ptr
                + phys_block * stride_ksb
                + kv_head * stride_ksh
                + offs_n * stride_ksn
            )
            k_scales = tl.load(k_scale_ptrs, mask=valid_mask, other=0.0).to(tl.float32)

            # Unpack nibbles: low = even index, high = odd index
            k_low = ((k_packed & 0x0F).to(tl.int32) - 8).to(tl.float32)
            k_high = (((k_packed >> 4) & 0x0F).to(tl.int32) - 8).to(tl.float32)

            dot_even = tl.sum(k_low * q_adj_even[None, :], 1)
            dot_odd = tl.sum(k_high * q_adj_odd[None, :], 1)
            scores = (dot_even + dot_odd) * k_scales * sm_scale
            scores = tl.where(valid_mask, scores, -float("inf"))

            m_curr = tl.max(scores, 0)
            m_new = tl.maximum(m_prev, m_curr)
            p = tl.exp(scores - m_new)
            p = tl.where(valid_mask, p, 0.0)

            alpha = tl.exp(m_prev - m_new)
            l_new = alpha * l_prev + tl.sum(p, 0)

            # Load packed INT4 V vectors (HALF_DIM bytes per token)
            v_ptrs = (
                V_pool_ptr
                + phys_block * stride_vb
                + kv_head * stride_vh
                + offs_n[:, None] * stride_vn
                + offs_half[None, :] * stride_vd
            )
            v_packed = tl.load(v_ptrs, mask=k_mask, other=0)

            # Load V scales: (BLOCK_SIZE,)
            v_scale_ptrs = (
                V_scale_ptr
                + phys_block * stride_vsb
                + kv_head * stride_vsh
                + offs_n * stride_vsn
            )
            v_scales = tl.load(v_scale_ptrs, mask=valid_mask, other=0.0).to(tl.float32)

            v_low = ((v_packed & 0x0F).to(tl.int32) - 8).to(tl.float32)
            v_high = (((v_packed >> 4) & 0x0F).to(tl.int32) - 8).to(tl.float32)

            p_scaled = p * v_scales
            acc_even = alpha * acc_even + tl.sum(p_scaled[:, None] * v_low, 0)
            acc_odd = alpha * acc_odd + tl.sum(p_scaled[:, None] * v_high, 0)

            m_prev = m_new
            l_prev = l_new

    if l_prev > 0.0:
        acc_even = acc_even / l_prev
        acc_odd = acc_odd / l_prev

    out_even_offset = cur_batch * stride_ob + cur_head * stride_oh + offs_even * stride_od
    out_odd_offset = cur_batch * stride_ob + cur_head * stride_oh + offs_odd * stride_od
    tl.store(Out_ptr + out_even_offset, acc_even.to(tl.float16))
    tl.store(Out_ptr + out_odd_offset, acc_odd.to(tl.float16))


def triton_adjoint_paged_attention_int4(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    k_scales: torch.Tensor,
    v_scales: torch.Tensor,
    block_tables: torch.Tensor,
    offset_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    sm_scale: Optional[float] = None,
    base: float = 10000.0,
    block_start_tables: Optional[torch.Tensor] = None,
    valid_lens_tables: Optional[torch.Tensor] = None,
    delta_bias: int = 16384,
) -> torch.Tensor:
    """Fused INT4 Adjoint PagedAttention with SRAM dequantization."""
    assert q.dtype == torch.float16, "q must be float16"
    assert k_pool.dtype == torch.uint8, "k_pool must be uint8 for packed INT4"
    assert v_pool.dtype == torch.uint8, "v_pool must be uint8 for packed INT4"

    batch_size, num_heads, q_len, head_dim = q.shape
    assert q_len == 1, "Decode PagedAttention expects q_len == 1"
    num_kv_heads = k_pool.shape[1]
    block_size = k_pool.shape[2]
    max_blocks = block_tables.shape[1]

    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)

    if block_start_tables is None:
        block_start_tables = (
            torch.arange(max_blocks, device=q.device, dtype=torch.int32) * block_size
        ).unsqueeze(0).expand(batch_size, -1).contiguous()
    if valid_lens_tables is None:
        valid_lens_tables = torch.clamp(
            seq_lens.unsqueeze(1) - block_start_tables, min=0, max=block_size
        ).to(torch.int32).contiguous()

    cos_cache, sin_cache, delta_bias_val = get_cos_sin_cache(
        head_dim=head_dim, base=base, max_delta=delta_bias, device=q.device
    )

    k_scales_s = k_scales.squeeze(-1) if k_scales.dim() == 4 else k_scales
    v_scales_s = v_scales.squeeze(-1) if v_scales.dim() == 4 else v_scales
    k_scales_s = k_scales_s.to(dtype=torch.float16)
    v_scales_s = v_scales_s.to(dtype=torch.float16)

    out = torch.empty_like(q)
    grid = (batch_size, num_heads)

    with torch.cuda.device(q.device):
        _triton_adjoint_paged_attn_int4_decode_kernel[grid](
            q, k_pool, v_pool, k_scales_s, v_scales_s, out,
            block_tables, offset_tables, block_start_tables, valid_lens_tables, seq_lens,
            cos_cache, sin_cache,
            sm_scale=sm_scale,
            delta_bias=delta_bias_val,
            stride_qb=q.stride(0), stride_qh=q.stride(1), stride_qm=q.stride(2), stride_qd=q.stride(3),
            stride_kb=k_pool.stride(0), stride_kh=k_pool.stride(1), stride_kn=k_pool.stride(2), stride_kd=k_pool.stride(3),
            stride_vb=v_pool.stride(0), stride_vh=v_pool.stride(1), stride_vn=v_pool.stride(2), stride_vd=v_pool.stride(3),
            stride_ksb=k_scales_s.stride(0), stride_ksh=k_scales_s.stride(1), stride_ksn=k_scales_s.stride(2),
            stride_vsb=v_scales_s.stride(0), stride_vsh=v_scales_s.stride(1), stride_vsn=v_scales_s.stride(2),
            stride_ob=out.stride(0), stride_oh=out.stride(1), stride_om=out.stride(2), stride_od=out.stride(3),
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

    return out
