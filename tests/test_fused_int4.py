import math
import pytest
import torch
import torch.nn.functional as F

from dag_kv.core.block_manager import BlockManager
from dag_kv.core.quantizer import KVQuantizer
from dag_kv.core.sink_canonicalizer import CanonicalChunk
from dag_kv.kernels import (
    adjoint_paged_attention,
    is_triton_available,
    pytorch_adjoint_paged_attention,
)
if is_triton_available():
    from dag_kv.kernels.triton_adjoint_paged_attn import triton_adjoint_paged_attention_int4


def test_block_manager_int4_allocation():
    bm = BlockManager(
        num_layers=4,
        num_kv_heads=2,
        head_dim=64,
        max_num_blocks=8,
        block_size=16,
        device="cpu",
        quantization="int4",
    )
    assert bm.k_pool.dtype == torch.uint8
    assert bm.v_pool.dtype == torch.uint8
    assert bm.k_pool.shape == (8, 4, 2, 16, 32)  # 64 // 2 = 32 bytes
    assert bm.v_pool.shape == (8, 4, 2, 16, 32)
    assert bm.k_scale_pool is not None
    assert bm.k_scale_pool.shape == (8, 4, 2, 16, 1)

    k_orig = torch.randn(4, 2, 16, 64, dtype=torch.float32)
    v_orig = torch.randn(4, 2, 16, 64, dtype=torch.float32)
    chunk = CanonicalChunk(
        chunk_id="int4_chunk",
        token_ids=torch.arange(16, dtype=torch.int64),
        k_cache=k_orig,
        v_cache=v_orig,
        lse_local=torch.zeros(2, 16, dtype=torch.float32),
        canonical_positions=torch.arange(4, 20, dtype=torch.int64),
        length=16,
    )
    b_ids = bm.register_canonical_chunk(chunk)
    assert len(b_ids) == 1
    # Check that pool data is populated
    assert not (bm.k_pool[b_ids[0]] == 0).all()


def test_pytorch_int4_fallback_cpu():
    torch.manual_seed(42)
    batch_size = 2
    num_heads = 4
    num_kv_heads = 2
    block_size = 16
    head_dim = 64
    seq_len = 16
    max_blocks = 4

    q = torch.randn(batch_size, num_heads, 1, head_dim, dtype=torch.float32)
    k_fp = torch.randn(max_blocks, num_kv_heads, block_size, head_dim, dtype=torch.float32)
    v_fp = torch.randn(max_blocks, num_kv_heads, block_size, head_dim, dtype=torch.float32)

    k_packed, k_scales = KVQuantizer.quantize_int4(k_fp, dim=-1)
    v_packed, v_scales = KVQuantizer.quantize_int4(v_fp, dim=-1)

    block_tables = torch.zeros(batch_size, 1, dtype=torch.int32)
    offset_tables = torch.zeros(batch_size, 1, dtype=torch.int32)
    seq_lens = torch.tensor([seq_len, seq_len], dtype=torch.int32)

    out = pytorch_adjoint_paged_attention(
        q=q,
        k_pool=k_packed,
        v_pool=v_packed,
        block_tables=block_tables,
        offset_tables=offset_tables,
        seq_lens=seq_lens,
        k_scales=k_scales,
        v_scales=v_scales,
    )
    assert out.shape == (batch_size, num_heads, 1, head_dim)
    assert not torch.isnan(out).any()


@pytest.mark.skipif(not torch.cuda.is_available() or not is_triton_available(), reason="CUDA + Triton required")
def test_fused_int4_triton_vs_pytorch_parity():
    torch.manual_seed(123)
    dev = torch.device("cuda:0")
    batch_size = 2
    num_heads = 4
    num_kv_heads = 2
    block_size = 16
    head_dim = 128
    seq_len = 16
    max_blocks = 4

    q = torch.randn(batch_size, num_heads, 1, head_dim, dtype=torch.float16, device=dev)
    k_fp = torch.randn(max_blocks, num_kv_heads, block_size, head_dim, dtype=torch.float16, device=dev)
    v_fp = torch.randn(max_blocks, num_kv_heads, block_size, head_dim, dtype=torch.float16, device=dev)

    k_packed, k_scales = KVQuantizer.quantize_int4(k_fp, dim=-1)
    v_packed, v_scales = KVQuantizer.quantize_int4(v_fp, dim=-1)

    block_tables = torch.zeros(batch_size, 1, dtype=torch.int32, device=dev)
    offset_tables = torch.zeros(batch_size, 1, dtype=torch.int32, device=dev)
    seq_lens = torch.tensor([seq_len, seq_len], dtype=torch.int32, device=dev)

    # PyTorch reference dequantizes then computes
    out_pt = pytorch_adjoint_paged_attention(
        q=q.float(),
        k_pool=k_packed,
        v_pool=v_packed,
        block_tables=block_tables,
        offset_tables=offset_tables,
        seq_lens=seq_lens,
        k_scales=k_scales.float(),
        v_scales=v_scales.float(),
    ).to(torch.float16)

    # Triton fused INT4 kernel runs in SRAM
    out_triton = triton_adjoint_paged_attention_int4(
        q=q,
        k_pool=k_packed,
        v_pool=v_packed,
        k_scales=k_scales,
        v_scales=v_scales,
        block_tables=block_tables,
        offset_tables=offset_tables,
        seq_lens=seq_lens,
    )

    cos_sim = F.cosine_similarity(out_pt.flatten().float(), out_triton.flatten().float(), dim=0).item()
    assert cos_sim > 0.98, f"Cosine similarity too low: {cos_sim}"
