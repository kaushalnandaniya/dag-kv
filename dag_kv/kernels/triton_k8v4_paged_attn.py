"""Fused K8V4 Adjoint PagedAttention Kernel (INT8 Keys + Group-32 INT4 Values)."""
import math
from typing import Optional
import torch
import triton
import triton.language as tl
from .triton_adjoint_paged_attn import get_cos_sin_cache


@triton.jit
def _k8v4_decode_kernel(
    Q_ptr, K_ptr, V_ptr, Ks_ptr, Vs_ptr, Out_ptr,
    Bt_ptr, Ot_ptr, Bs_ptr, Vl_ptr, Sl_ptr, Cos_ptr, Sin_ptr,
    sm_scale: tl.constexpr, delta_bias: tl.constexpr,
    sqb, sqh, sqm, sqd, skb, skh, skn, skd, svb, svh, svn, svd,
    sksb, sksh, sksn, svsb, svsh, svsn, svsg, sob, soh, som, sod,
    sbtb, sbtn, sotb, sotn, sbsb, sbsn, svlb, svln, scd, scdim, ssd, ssdim,
    num_total_blocks: int, num_heads: tl.constexpr, num_kv_heads: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr, V_GROUP_HALF_DIM: tl.constexpr,
):
    cb = tl.program_id(0)
    ch = tl.program_id(1)
    kvh = ch // (num_heads // num_kv_heads)
    seq_len = tl.load(Sl_ptr + cb)
    if seq_len <= 0:
        return

    HALF: tl.constexpr = HEAD_DIM // 2
    od = tl.arange(0, HEAD_DIM)
    ridx = (od + HALF) % HEAD_DIM
    rsgn = tl.where(od < HALF, -1.0, 1.0).to(tl.float32)
    q_vec = tl.load(Q_ptr + cb * sqb + ch * sqh + od * sqd).to(tl.float32)
    q_rot = tl.load(Q_ptr + cb * sqb + ch * sqh + ridx * sqd).to(tl.float32) * rsgn

    oh = tl.arange(0, HALF)
    vg = oh // V_GROUP_HALF_DIM
    oe = oh * 2
    oo = oh * 2 + 1
    on = tl.arange(0, BLOCK_SIZE)

    m_prev = -float("inf")
    l_prev = 0.0
    acc_e = tl.zeros([HALF], dtype=tl.float32)
    acc_o = tl.zeros([HALF], dtype=tl.float32)

    for b in range(num_total_blocks):
        bstart = tl.load(Bs_ptr + cb * sbsb + b * sbsn)
        vlen = tl.load(Vl_ptr + cb * svlb + b * svln)
        if bstart < seq_len and vlen > 0:
            pb = tl.load(Bt_ptr + cb * sbtb + b * sbtn)
            drow = tl.load(Ot_ptr + cb * sotb + b * sotn) + delta_bias
            cd = tl.load(Cos_ptr + drow * scd + od * scdim).to(tl.float32)
            sd = tl.load(Sin_ptr + drow * ssd + od * ssdim).to(tl.float32)
            q_adj = q_vec * cd - q_rot * sd

            vm = (on < vlen) & ((bstart + on) < seq_len)
            km = vm[:, None]
            k8 = tl.load(K_ptr + pb * skb + kvh * skh + on[:, None] * skn + od[None, :] * skd, mask=km, other=0).to(tl.float32)
            ks = tl.load(Ks_ptr + pb * sksb + kvh * sksh + on * sksn, mask=vm, other=0.0).to(tl.float32)

            sc = tl.where(vm, tl.sum(k8 * q_adj[None, :], 1) * ks * sm_scale, -float("inf"))
            m_new = tl.maximum(m_prev, tl.max(sc, 0))
            p = tl.where(vm, tl.exp(sc - m_new), 0.0)
            alpha = tl.exp(m_prev - m_new)
            l_new = alpha * l_prev + tl.sum(p, 0)

            vp = tl.load(V_ptr + pb * svb + kvh * svh + on[:, None] * svn + oh[None, :] * svd, mask=km, other=0).to(tl.int32)
            ve = ((vp & 0x0F) - 8).to(tl.float32)
            vo = (((vp >> 4) & 0x0F) - 8).to(tl.float32)
            vs = tl.load(Vs_ptr + pb * svsb + kvh * svsh + on[:, None] * svsn + vg[None, :] * svsg, mask=km, other=0.0).to(tl.float32)

            ps = p[:, None] * vs
            acc_e = alpha * acc_e + tl.sum(ps * ve, 0)
            acc_o = alpha * acc_o + tl.sum(ps * vo, 0)
            m_prev = m_new
            l_prev = l_new

    if l_prev > 0.0:
        acc_e = acc_e / l_prev
        acc_o = acc_o / l_prev
    ob = Out_ptr + cb * sob + ch * soh
    tl.store(ob + oe * sod, acc_e.to(tl.float16))
    tl.store(ob + oo * sod, acc_o.to(tl.float16))


def triton_adjoint_paged_attention_k8v4(
    q: torch.Tensor, k_pool: torch.Tensor, v_pool: torch.Tensor,
    k_scales: torch.Tensor, v_scales: torch.Tensor,
    block_tables: torch.Tensor, offset_tables: torch.Tensor, seq_lens: torch.Tensor,
    sm_scale: Optional[float] = None, base: float = 10000.0,
    block_start_tables: Optional[torch.Tensor] = None, valid_lens_tables: Optional[torch.Tensor] = None,
    delta_bias: int = 32768,
) -> torch.Tensor:
    assert q.is_cuda and k_pool.dtype == torch.int8 and v_pool.dtype == torch.uint8
    orig = q.dtype
    qf = q if q.dtype == torch.float16 else q.to(torch.float16)
    B, H, _, D = qf.shape
    _, KVH, BS, _ = k_pool.shape
    M = block_tables.shape[1]
    sm_scale = sm_scale if sm_scale is not None else 1.0 / math.sqrt(D)
    bt = block_tables.to(torch.int32).contiguous()
    ot = offset_tables.to(torch.int32).contiguous()
    sl = seq_lens.to(torch.int32).contiguous()
    bs = (torch.arange(M, device=q.device, dtype=torch.int32).unsqueeze(0).expand(B, -1) * BS).contiguous() if block_start_tables is None else block_start_tables.to(torch.int32).contiguous()
    vl = torch.clamp(sl.unsqueeze(1) - bs, min=0, max=BS).to(torch.int32).contiguous() if valid_lens_tables is None else valid_lens_tables.to(torch.int32).contiguous()
    cc, sc, db = get_cos_sin_cache(head_dim=D, base=base, max_delta=delta_bias, device=q.device)
    ks = (k_scales.squeeze(-1) if k_scales.dim() == 4 else k_scales).to(torch.float16)
    vs = (v_scales.unsqueeze(-1) if v_scales.dim() == 3 else v_scales).to(torch.float16)
    out = torch.empty_like(qf)
    with torch.cuda.device(q.device):
        _k8v4_decode_kernel[(B, H)](
            qf, k_pool, v_pool, ks, vs, out, bt, ot, bs, vl, sl, cc, sc,
            sm_scale=sm_scale, delta_bias=db,
            sqb=qf.stride(0), sqh=qf.stride(1), sqm=qf.stride(2), sqd=qf.stride(3),
            skb=k_pool.stride(0), skh=k_pool.stride(1), skn=k_pool.stride(2), skd=k_pool.stride(3),
            svb=v_pool.stride(0), svh=v_pool.stride(1), svn=v_pool.stride(2), svd=v_pool.stride(3),
            sksb=ks.stride(0), sksh=ks.stride(1), sksn=ks.stride(2),
            svsb=vs.stride(0), svsh=vs.stride(1), svsn=vs.stride(2), svsg=vs.stride(3),
            sob=out.stride(0), soh=out.stride(1), som=out.stride(2), sod=out.stride(3),
            sbtb=bt.stride(0), sbtn=bt.stride(1), sotb=ot.stride(0), sotn=ot.stride(1),
            sbsb=bs.stride(0), sbsn=bs.stride(1), svlb=vl.stride(0), svln=vl.stride(1),
            scd=cc.stride(0), scdim=cc.stride(1), ssd=sc.stride(0), ssdim=sc.stride(1),
            num_total_blocks=M, num_heads=H, num_kv_heads=KVH, BLOCK_SIZE=BS, HEAD_DIM=D,
            V_GROUP_HALF_DIM=(D // 2) // vs.shape[-1], num_warps=4, num_stages=2,
        )
    return out if out.dtype == orig else out.to(orig)
