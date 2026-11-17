"""Tests for Adaptive KV Cache Quantization (INT8 & INT4)."""

import pytest
import torch

from dag_kv.core.block_manager import BlockManager
from dag_kv.core.quantizer import KVQuantizer
from dag_kv.core.sink_canonicalizer import CanonicalChunk


def test_kv_quantizer_int8_accuracy():
    """Verifies that symmetric INT8 quantization maintains low reconstruction error."""
    torch.manual_seed(42)
    x = torch.randn(4, 2, 16, 64, dtype=torch.float32)

    x_q, scale = KVQuantizer.quantize_int8(x, dim=-1)
    assert x_q.dtype == torch.int8
    assert x_q.shape == x.shape
    assert scale.shape == (4, 2, 16, 1)

    x_recon = KVQuantizer.dequantize_int8(x_q, scale)
    rel_error = (torch.abs(x - x_recon) / (torch.abs(x) + 1e-5)).mean().item()
    # INT8 quantization relative error is typically < 2%
    assert rel_error < 0.05, f"INT8 relative error too high: {rel_error}"


def test_kv_quantizer_int4_accuracy():
    """Verifies that 4-bit packed quantization compresses memory by 75%."""
    torch.manual_seed(42)
    x = torch.randn(4, 2, 16, 64, dtype=torch.float32)

    x_packed, scale = KVQuantizer.quantize_int4(x, dim=-1)
    assert x_packed.dtype == torch.uint8
    # Packed dimension is halved (64 / 2 = 32 bytes)
    assert x_packed.shape == (4, 2, 16, 32)

    x_recon = KVQuantizer.dequantize_int4(x_packed, scale)
    assert x_recon.shape == x.shape
    cos_sim = torch.cosine_similarity(x.flatten(), x_recon.flatten(), dim=0).item()
    assert cos_sim > 0.95, f"INT4 cosine similarity too low: {cos_sim}"


def test_block_manager_int8_quantization_lifecycle():
    """Validates INT8 BlockManager allocation, chunk storage, and CoW copying."""
    num_layers = 2
    num_kv_heads = 2
    head_dim = 64
    block_size = 16
    chunk_len = 32

    bm = BlockManager(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_num_blocks=16,
        block_size=block_size,
        device="cpu",
        dtype=torch.float32,
        quantization="int8",
    )

    assert bm.k_pool.dtype == torch.int8
    assert bm.v_pool.dtype == torch.int8
    assert bm.k_scale_pool is not None
    assert bm.v_scale_pool is not None

    # Register chunk
    token_ids = torch.arange(chunk_len, dtype=torch.int64)
    k_cache = torch.randn(num_layers, num_kv_heads, chunk_len, head_dim)
    v_cache = torch.randn(num_layers, num_kv_heads, chunk_len, head_dim)
    lse = torch.zeros(num_layers, num_kv_heads, chunk_len)

    chunk = CanonicalChunk(
        chunk_id="test_int8_chunk",
        token_ids=token_ids,
        k_cache=k_cache,
        v_cache=v_cache,
        lse_local=lse,
        canonical_positions=torch.arange(chunk_len),
        length=chunk_len,
    )

    phys_blocks = bm.register_canonical_chunk(chunk)
    assert len(phys_blocks) == 2

    # Verify physical blocks hold non-zero int8 values
    assert bm.k_pool[phys_blocks[0]].abs().sum() > 0

    # Test CoW shadow block creation preserving scales
    table = bm.assemble_dag_sequence("seq_int8", [(chunk, 4)])
    shadow_id = bm.create_cow_shadow_block(table, 0)
    assert shadow_id != phys_blocks[0]
    assert bm.k_pool[shadow_id].dtype == torch.int8
    assert (bm.k_scale_pool[shadow_id] == bm.k_scale_pool[phys_blocks[0]]).all()
