"""Fused Triton KV Quantize-and-Scatter Write Kernel (INT8, K8V4, INT4 K8g/V32g)."""
import torch
import triton
import triton.language as tl


@triton.jit
def _fused_kv_quant_write_kernel(
    Kin_ptr, Vin_ptr, Kpool_ptr, Vpool_ptr, Ksc_ptr, Vsc_ptr, Phys_ptr, Slot_ptr,
    skin_n, skin_h, skin_d, svin_n, svin_h, svin_d,
    skp_b, skp_h, skp_s, skp_d, svp_b, svp_h, svp_s, svp_d,
    sks_b, sks_h, sks_s, sks_g, svs_b, svs_h, svs_s, svs_g,
    HEAD_DIM: tl.constexpr,
    K_IS_INT4: tl.constexpr, V_IS_INT4: tl.constexpr,
    K_GROUP: tl.constexpr, V_GROUP: tl.constexpr,
):
    n = tl.program_id(0)
    h = tl.program_id(1)
    pb = tl.load(Phys_ptr + n).to(tl.int64)
    sl = tl.load(Slot_ptr + n).to(tl.int64)

    kin_base = Kin_ptr + n * skin_n + h * skin_h
    kp_base = Kpool_ptr + pb * skp_b + h * skp_h + sl * skp_s
    ks_base = Ksc_ptr + pb * sks_b + h * sks_h + sl * sks_s

    if K_IS_INT4:
        K_NG: tl.constexpr = HEAD_DIM // K_GROUP
        K_HG: tl.constexpr = K_GROUP // 2
        kg = tl.arange(0, K_NG)
        khg = tl.arange(0, K_HG)
        ke_d = kg[:, None] * K_GROUP + khg[None, :] * 2
        xe = tl.load(kin_base + ke_d * skin_d).to(tl.float32)
        xo = tl.load(kin_base + (ke_d + 1) * skin_d).to(tl.float32)
        k_sc = tl.maximum(tl.max(tl.maximum(tl.abs(xe), tl.abs(xo)), axis=1) / 7.0, 1e-8)
        qe = xe / k_sc[:, None]
        qo = xo / k_sc[:, None]
        qei = tl.where(qe >= 0.0, tl.floor(qe + 0.5), -tl.floor(-qe + 0.5)).to(tl.int32)
        qoi = tl.where(qo >= 0.0, tl.floor(qo + 0.5), -tl.floor(-qo + 0.5)).to(tl.int32)
        qeu = (tl.maximum(tl.minimum(qei, 7), -8) + 8) & 0x0F
        qou = (tl.maximum(tl.minimum(qoi, 7), -8) + 8) & 0x0F
        k_pack = (qeu | (qou << 4)).to(tl.uint8)
        k_pd = kg[:, None] * K_HG + khg[None, :]
        tl.store(kp_base + k_pd * skp_d, k_pack)
        tl.store(ks_base + kg * sks_g, k_sc.to(tl.float16))
    else:
        od = tl.arange(0, HEAD_DIM)
        xk = tl.load(kin_base + od * skin_d).to(tl.float32)
        k_sc = tl.maximum(tl.max(tl.abs(xk), axis=0) / 127.0, 1e-8)
        qk = xk / k_sc
        qki = tl.where(qk >= 0.0, tl.floor(qk + 0.5), -tl.floor(-qk + 0.5)).to(tl.int32)
        tl.store(kp_base + od * skp_d, tl.maximum(tl.minimum(qki, 127), -128).to(tl.int8))
        tl.store(ks_base, k_sc.to(tl.float16))

    vin_base = Vin_ptr + n * svin_n + h * svin_h
    vp_base = Vpool_ptr + pb * svp_b + h * svp_h + sl * svp_s
    vs_base = Vsc_ptr + pb * svs_b + h * svs_h + sl * svs_s

    if V_IS_INT4:
        V_NG: tl.constexpr = HEAD_DIM // V_GROUP
        V_HG: tl.constexpr = V_GROUP // 2
        vg = tl.arange(0, V_NG)
        vhg = tl.arange(0, V_HG)
        ve_d = vg[:, None] * V_GROUP + vhg[None, :] * 2
        ye = tl.load(vin_base + ve_d * svin_d).to(tl.float32)
        yo = tl.load(vin_base + (ve_d + 1) * svin_d).to(tl.float32)
        v_sc = tl.maximum(tl.max(tl.maximum(tl.abs(ye), tl.abs(yo)), axis=1) / 7.0, 1e-8)
        qe = ye / v_sc[:, None]
        qo = yo / v_sc[:, None]
        qei = tl.where(qe >= 0.0, tl.floor(qe + 0.5), -tl.floor(-qe + 0.5)).to(tl.int32)
        qoi = tl.where(qo >= 0.0, tl.floor(qo + 0.5), -tl.floor(-qo + 0.5)).to(tl.int32)
        qeu = (tl.maximum(tl.minimum(qei, 7), -8) + 8) & 0x0F
        qou = (tl.maximum(tl.minimum(qoi, 7), -8) + 8) & 0x0F
        v_pack = (qeu | (qou << 4)).to(tl.uint8)
        v_pd = vg[:, None] * V_HG + vhg[None, :]
        tl.store(vp_base + v_pd * svp_d, v_pack)
        tl.store(vs_base + vg * svs_g, v_sc.to(tl.float16))
    else:
        od = tl.arange(0, HEAD_DIM)
        xv = tl.load(vin_base + od * svin_d).to(tl.float32)
        v_sc = tl.maximum(tl.max(tl.abs(xv), axis=0) / 127.0, 1e-8)
        qv = xv / v_sc
        qvi = tl.where(qv >= 0.0, tl.floor(qv + 0.5), -tl.floor(-qv + 0.5)).to(tl.int32)
        tl.store(vp_base + od * svp_d, tl.maximum(tl.minimum(qvi, 127), -128).to(tl.int8))
        tl.store(vs_base, v_sc.to(tl.float16))


def triton_fused_kv_quant_write(
    k_in: torch.Tensor, v_in: torch.Tensor,
    k_pool_l: torch.Tensor, v_pool_l: torch.Tensor,
    k_scale_l: torch.Tensor, v_scale_l: torch.Tensor,
    phys_ids: torch.Tensor, slot_offs: torch.Tensor,
    k_group_size: int = 8, v_group_size: int = 32,
) -> None:
    """Fused quantize + pack + scatter for (N, num_kv_heads, head_dim) into layer pool views."""
    n_tok, num_kv_heads, head_dim = k_in.shape
    k_is_int4 = k_pool_l.dtype == torch.uint8
    v_is_int4 = v_pool_l.dtype == torch.uint8
    with torch.cuda.device(k_in.device):
        _fused_kv_quant_write_kernel[(n_tok, num_kv_heads)](
            k_in, v_in, k_pool_l, v_pool_l, k_scale_l, v_scale_l, phys_ids, slot_offs,
            k_in.stride(0), k_in.stride(1), k_in.stride(2),
            v_in.stride(0), v_in.stride(1), v_in.stride(2),
            k_pool_l.stride(0), k_pool_l.stride(1), k_pool_l.stride(2), k_pool_l.stride(3),
            v_pool_l.stride(0), v_pool_l.stride(1), v_pool_l.stride(2), v_pool_l.stride(3),
            k_scale_l.stride(0), k_scale_l.stride(1), k_scale_l.stride(2), k_scale_l.stride(3),
            v_scale_l.stride(0), v_scale_l.stride(1), v_scale_l.stride(2), v_scale_l.stride(3),
            HEAD_DIM=head_dim, K_IS_INT4=k_is_int4, V_IS_INT4=v_is_int4,
            K_GROUP=k_group_size if k_is_int4 else 8,
            V_GROUP=v_group_size if v_is_int4 else 32,
            num_warps=2,
        )
