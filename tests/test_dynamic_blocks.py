"""Tests for Dynamic and Configurable Block Sizes (8, 16, 32)."""

import pytest
import torch

from dag_kv.core.block_manager import BlockManager
from dag_kv.core.sink_canonicalizer import CanonicalChunk
from dag_kv.kernels.pytorch_fallbacks import pytorch_adjoint_paged_attention


@pytest.mark.parametrize("block_size", [8, 16, 32])
def test_dynamic_block_size_allocation(block_size):
    """Verifies that BlockManager correctly partitions chunks according to block_size."""
    num_layers = 1
    num_kv_heads = 2
    head_dim = 64
    chunk_len = 50  # Not a multiple of 8, 16, or 32

    bm = BlockManager(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        max_num_blocks=32,
        block_size=block_size,
        device="cpu",
        dtype=torch.float32,
    )

    k_cache = torch.randn(num_layers, num_kv_heads, chunk_len, head_dim)
    v_cache = torch.randn(num_layers, num_kv_heads, chunk_len, head_dim)

    chunk = CanonicalChunk(
        chunk_id=f"chunk_bs_{block_size}",
        token_ids=torch.arange(chunk_len),
        k_cache=k_cache,
        v_cache=v_cache,
        lse_local=torch.zeros(num_layers, num_kv_heads, chunk_len),
        canonical_positions=torch.arange(chunk_len),
        length=chunk_len,
    )

    phys_blocks = bm.register_canonical_chunk(chunk)
    expected_blocks = (chunk_len + block_size - 1) // block_size
    assert len(phys_blocks) == expected_blocks

    table = bm.assemble_dag_sequence("seq_dyn", [(chunk, 4)])
    assert len(table.blocks) == expected_blocks
    total_valid = sum(b.valid_tokens for b in table.blocks)
    assert total_valid == chunk_len


@pytest.mark.parametrize("block_size", [8, 32])
def test_attention_with_variable_block_sizes(block_size):
    """Verifies that PyTorch paged attention executes correctly on non-16 block sizes."""
    batch_size = 1
    num_heads = 4
    num_kv_heads = 2
    head_dim = 64
    seq_len = 48
    num_blocks = (seq_len + block_size - 1) // block_size

    q = torch.randn(batch_size, num_heads, 1, head_dim)
    k_pool = torch.randn(num_blocks, num_kv_heads, block_size, head_dim)
    v_pool = torch.randn(num_blocks, num_kv_heads, block_size, head_dim)

    block_tables = torch.arange(num_blocks, dtype=torch.int32).unsqueeze(0)
    offset_tables = torch.zeros((batch_size, num_blocks), dtype=torch.int32)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32)

    out = pytorch_adjoint_paged_attention(
        q=q,
        k_pool=k_pool,
        v_pool=v_pool,
        block_tables=block_tables,
        offset_tables=offset_tables,
        seq_lens=seq_lens,
    )

    assert out.shape == (batch_size, num_heads, 1, head_dim)
    assert not torch.isnan(out).any()
