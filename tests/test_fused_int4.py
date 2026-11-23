import pytest
import torch
import torch.nn.functional as F

from dag_kv.core.block_manager import BlockManager
from dag_kv.core.quantizer import KVQuantizer
from dag_kv.core.sink_canonicalizer import CanonicalChunk
from dag_kv.kernels import is_triton_available, pytorch_adjoint_paged_attention
from dag_kv.kernels.dispatch import adjoint_paged_attention, sparse_bridge_attention
from dag_kv.kernels.pytorch_fallbacks import pytorch_sparse_bridge_attention


def _make_chunk(cid: str, head_dim: int = 64) -> CanonicalChunk:
    return CanonicalChunk(
        chunk_id=cid,
        token_ids=torch.arange(16, dtype=torch.int64),
        k_cache=torch.randn(4, 2, 16, head_dim, dtype=torch.float32),
        v_cache=torch.randn(4, 2, 16, head_dim, dtype=torch.float32),
        lse_local=torch.zeros(2, 16, dtype=torch.float32),
        canonical_positions=torch.arange(4, 20, dtype=torch.int64),
        length=16,
    )


def test_block_manager_int4_allocation():
    bm = BlockManager(4, 2, 64, max_num_blocks=8, block_size=16, device="cpu", quantization="int4")
    assert bm.k_pool.dtype == torch.uint8 and bm.v_pool.dtype == torch.uint8
    assert bm.k_pool.shape == (8, 4, 2, 16, 32)
    assert bm.v_pool.shape == (8, 4, 2, 16, 32)
    assert bm.k_scale_pool.shape == (8, 4, 2, 16, 8)  # K=g8
    assert bm.v_scale_pool.shape == (8, 4, 2, 16, 2)  # V=g32
    b_ids = bm.register_canonical_chunk(_make_chunk("int4_chunk"))
    assert len(b_ids) == 1 and not (bm.k_pool[b_ids[0]] == 0).all()


def test_block_manager_k8v4_allocation():
    bm = BlockManager(4, 2, 64, max_num_blocks=8, block_size=16, device="cpu", quantization="k8v4")
    assert bm.k_pool.dtype == torch.int8 and bm.v_pool.dtype == torch.uint8
    assert bm.k_pool.shape == (8, 4, 2, 16, 64)
    assert bm.v_pool.shape == (8, 4, 2, 16, 32)
    assert bm.k_scale_pool.shape == (8, 4, 2, 16, 1)
    assert bm.v_scale_pool.shape == (8, 4, 2, 16, 2)
    b_ids = bm.register_canonical_chunk(_make_chunk("k8v4_chunk"))
    assert len(b_ids) == 1 and not (bm.k_pool[b_ids[0]] == 0).all() and not (bm.v_pool[b_ids[0]] == 0).all()


@pytest.mark.parametrize("mode", ["int4", "k8v4"])
def test_pytorch_quant_fallback_cpu(mode):
    torch.manual_seed(42)
    B, H, KVH, BS, D = 2, 4, 2, 16, 64
    q = torch.randn(B, H, 1, D, dtype=torch.float32)
    k_fp = torch.randn(4, KVH, BS, D, dtype=torch.float32)
    v_fp = torch.randn(4, KVH, BS, D, dtype=torch.float32)

    if mode == "k8v4":
        kp, ks = KVQuantizer.quantize_int8(k_fp, dim=-1)
    else:
        kp, ks = KVQuantizer.quantize_int4(k_fp, dim=-1, group_size=KVQuantizer.INT4_K_GROUP_SIZE)
    vp, vs = KVQuantizer.quantize_int4(v_fp, dim=-1, group_size=KVQuantizer.INT4_V_GROUP_SIZE)

    bt = torch.zeros(B, 1, dtype=torch.int32)
    ot = torch.zeros(B, 1, dtype=torch.int32)
    sl = torch.tensor([16, 16], dtype=torch.int32)
    out = pytorch_adjoint_paged_attention(q, kp, vp, bt, ot, sl, k_scales=ks, v_scales=vs)
    assert out.shape == (B, H, 1, D) and not torch.isnan(out).any()

    qb = torch.randn(B, H, 3, D, dtype=torch.float32)
    bpos = torch.tensor([[3, 7, 15], [2, 8, 14]], dtype=torch.int32)
    out_b = pytorch_sparse_bridge_attention(qb, bpos, kp, vp, bt, ot, k_scales=ks, v_scales=vs)
    assert out_b.shape == (B, H, 3, D) and not torch.isnan(out_b).any()


@pytest.mark.skipif(not torch.cuda.is_available() or not is_triton_available(), reason="CUDA + Triton required")
@pytest.mark.parametrize("mode", ["int8", "k8v4", "int4"])
def test_fused_quant_triton_vs_pytorch_parity(mode):
    torch.manual_seed(123)
    dev = torch.device("cuda:0")
    B, H, KVH, BS, D = 2, 8, 2, 16, 128
    q = torch.randn(B, H, 1, D, dtype=torch.float16, device=dev)
    k_fp = torch.randn(6, KVH, BS, D, dtype=torch.float16, device=dev)
    v_fp = torch.randn(6, KVH, BS, D, dtype=torch.float16, device=dev)

    if mode == "int8":
        kp, ks = KVQuantizer.quantize_int8(k_fp, dim=-1)
        vp, vs = KVQuantizer.quantize_int8(v_fp, dim=-1)
    elif mode == "k8v4":
        kp, ks = KVQuantizer.quantize_int8(k_fp, dim=-1)
        vp, vs = KVQuantizer.quantize_int4(v_fp, dim=-1, group_size=KVQuantizer.INT4_V_GROUP_SIZE)
    else:
        kp, ks = KVQuantizer.quantize_int4(k_fp, dim=-1, group_size=KVQuantizer.INT4_K_GROUP_SIZE)
        vp, vs = KVQuantizer.quantize_int4(v_fp, dim=-1, group_size=KVQuantizer.INT4_V_GROUP_SIZE)

    bt = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32, device=dev)
    ot = torch.tensor([[0, 12, -4], [0, -16, 32]], dtype=torch.int32, device=dev)
    bs = torch.tensor([[0, 4, 20], [0, 4, 20]], dtype=torch.int32, device=dev)
    vl = torch.tensor([[4, 16, 10], [4, 16, 12]], dtype=torch.int32, device=dev)
    sl = torch.tensor([30, 32], dtype=torch.int32, device=dev)

    out_tr = adjoint_paged_attention(q, kp, vp, bt, ot, sl, base=1e6, block_start_tables=bs, valid_lens_tables=vl, k_scales=ks, v_scales=vs)
    out_pt = pytorch_adjoint_paged_attention(q.float(), kp, vp, bt, ot, sl, base=1e6, block_start_tables=bs, valid_lens_tables=vl, k_scales=ks.float(), v_scales=vs.float()).to(torch.float16)
    cos_dec = F.cosine_similarity(out_pt.flatten().float(), out_tr.flatten().float(), dim=0).item()
    assert cos_dec > 0.999, f"[{mode}] decode parity failed: cos={cos_dec}"

    qb = torch.randn(B, H, 3, D, dtype=torch.float16, device=dev)
    bpos = torch.tensor([[4, 15, 28], [3, 18, 30]], dtype=torch.int32, device=dev)
    br_tr = sparse_bridge_attention(qb, bpos, kp, vp, bt, ot, base=1e6, block_start_tables=bs, valid_lens_tables=vl, k_scales=ks, v_scales=vs)
    br_pt = pytorch_sparse_bridge_attention(qb.float(), bpos, kp, vp, bt, ot, base=1e6, block_start_tables=bs, valid_lens_tables=vl, k_scales=ks.float(), v_scales=vs.float()).to(torch.float16)
    cos_br = F.cosine_similarity(br_pt.flatten().float(), br_tr.flatten().float(), dim=0).item()
    assert cos_br > 0.999, f"[{mode}] bridge parity failed: cos={cos_br}"


@pytest.mark.skipif(not torch.cuda.is_available() or not is_triton_available(), reason="CUDA + Triton required")
@pytest.mark.parametrize("mode", ["int8", "k8v4", "int4"])
def test_fused_kv_quant_write_parity(mode):
    from dag_kv.kernels.dispatch import fused_kv_quant_write

    torch.manual_seed(7)
    dev = torch.device("cuda:0")
    bm = BlockManager(1, 4, 128, max_num_blocks=8, block_size=16, device=dev, quantization=mode)
    k_in = torch.randn(3, 4, 128, dtype=torch.float16, device=dev)
    v_in = torch.randn(3, 4, 128, dtype=torch.float16, device=dev)
    phys = torch.tensor([1, 4, 6], dtype=torch.long, device=dev)
    slots = torch.tensor([2, 9, 15], dtype=torch.long, device=dev)

    fused_kv_quant_write(
        k_in, v_in, bm.k_pool[:, 0], bm.v_pool[:, 0],
        bm.k_scale_pool[:, 0], bm.v_scale_pool[:, 0], phys, slots, mode,
    )
    k_ref_q, k_ref_s = (
        KVQuantizer.quantize_int4(k_in, dim=-1, group_size=KVQuantizer.INT4_K_GROUP_SIZE)
        if mode == "int4" else KVQuantizer.quantize_int8(k_in, dim=-1)
    )
    v_ref_q, v_ref_s = (
        KVQuantizer.quantize_int4(v_in, dim=-1, group_size=KVQuantizer.INT4_V_GROUP_SIZE)
        if mode in ("k8v4", "int4") else KVQuantizer.quantize_int8(v_in, dim=-1)
    )
    k_tr = (
        KVQuantizer.dequantize_int4(bm.k_pool[phys, 0, :, slots, :], bm.k_scale_pool[phys, 0, :, slots, :], torch.float32)
        if mode == "int4" else bm.k_pool[phys, 0, :, slots, :].float() * bm.k_scale_pool[phys, 0, :, slots, :].float()
    )
    k_pt = (
        KVQuantizer.dequantize_int4(k_ref_q, k_ref_s, torch.float32)
        if mode == "int4" else k_ref_q.float() * k_ref_s.float()
    )
    v_tr = (
        KVQuantizer.dequantize_int4(bm.v_pool[phys, 0, :, slots, :], bm.v_scale_pool[phys, 0, :, slots, :], torch.float32)
        if mode in ("k8v4", "int4") else bm.v_pool[phys, 0, :, slots, :].float() * bm.v_scale_pool[phys, 0, :, slots, :].float()
    )
    v_pt = (
        KVQuantizer.dequantize_int4(v_ref_q, v_ref_s, torch.float32)
        if mode in ("k8v4", "int4") else v_ref_q.float() * v_ref_s.float()
    )
    assert F.cosine_similarity(k_tr.flatten(), k_pt.flatten(), dim=0).item() > 0.9999
    assert F.cosine_similarity(v_tr.flatten(), v_pt.flatten(), dim=0).item() > 0.9999
