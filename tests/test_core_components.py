"""Unit tests for SinkCanonicalizer, LSEProbe, and BlockManager."""

import pytest
import torch
from dag_kv.core.sink_canonicalizer import SinkCanonicalizer, CanonicalChunk
from dag_kv.core.lse_probe import LSEProbe, compute_lse_perturbation, select_bridge_tokens
from dag_kv.core.block_manager import BlockManager, OffsetBlockTable


def test_sink_canonicalizer():
    canon = SinkCanonicalizer(num_sink_tokens=4)
    tokens = torch.tensor([101, 102, 103, 104, 105], dtype=torch.int64)

    input_ids, attn_mask, pos_ids = canon.prepare_prefill_input(tokens)
    assert input_ids.shape == (1, 9)  # 4 sink + 5 tokens
    assert attn_mask.shape == (1, 9)
    assert pos_ids.shape == (1, 9)
    assert pos_ids[0, 0].item() == 0
    assert pos_ids[0, 4].item() == 4

    # Dummy KV caches
    layers = 2
    heads = 2
    head_dim = 64
    k = torch.randn(layers, heads, 9, head_dim)
    v = torch.randn(layers, heads, 9, head_dim)
    lse = torch.randn(layers, heads, 9)

    chunk = canon.canonicalize_kv(k, v, lse, tokens)
    assert chunk.length == 5
    assert chunk.k_cache.shape == (layers, heads, 5, head_dim)
    assert chunk.v_cache.shape == (layers, heads, 5, head_dim)
    assert chunk.lse_local.shape == (layers, heads, 5)
    assert chunk.canonical_positions.tolist() == [4, 5, 6, 7, 8]
    assert chunk.k_cache.dtype == torch.float16
    assert chunk.v_cache.dtype == torch.float16


def test_lse_probe_and_bridge_selection():
    torch.manual_seed(42)
    num_heads = 4
    num_kv_heads = 2
    l_target = 100
    l_prior = 50
    head_dim = 64

    q = torch.randn(num_heads, l_target, head_dim)
    k_prior = torch.randn(num_kv_heads, l_prior, head_dim)
    lse_local = torch.randn(num_heads, l_target)

    alpha = compute_lse_perturbation(q, k_prior, lse_local)
    assert alpha.shape == (l_target,)
    assert torch.all(alpha >= 0.0) and torch.all(alpha <= 1.0)

    # Bridge selection with 8% top-k
    result = select_bridge_tokens(alpha, top_k_percent=0.08, chunk_global_offset=200)
    assert result.num_bridge_tokens == 8
    assert len(result.bridge_indices) == 8
    assert len(result.global_bridge_indices) == 8
    assert (result.global_bridge_indices == result.bridge_indices + 200).all()
    # Check that indices are strictly sorted
    assert torch.equal(result.bridge_indices, torch.sort(result.bridge_indices)[0])


def test_block_manager_cow_and_assembly():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mgr = BlockManager(
        num_layers=2,
        num_kv_heads=2,
        head_dim=64,
        max_num_blocks=32,
        block_size=16,
        device=device,
        dtype=torch.float16,
    )

    canon = SinkCanonicalizer(num_sink_tokens=4)
    tokens_a = torch.arange(10, 42, dtype=torch.int64)  # 32 tokens -> 2 blocks
    k_a = torch.randn(2, 2, 36, 64, device=device, dtype=torch.float16)
    v_a = torch.randn(2, 2, 36, 64, device=device, dtype=torch.float16)
    lse_a = torch.randn(2, 2, 36, device=device, dtype=torch.float32)
    chunk_a = canon.canonicalize_kv(k_a, v_a, lse_a, tokens_a)

    tokens_b = torch.arange(100, 116, dtype=torch.int64)  # 16 tokens -> 1 block
    k_b = torch.randn(2, 2, 20, 64, device=device, dtype=torch.float16)
    v_b = torch.randn(2, 2, 20, 64, device=device, dtype=torch.float16)
    lse_b = torch.randn(2, 2, 20, device=device, dtype=torch.float32)
    chunk_b = canon.canonicalize_kv(k_b, v_b, lse_b, tokens_b)

    # Assemble sequence: chunk_a at 4, chunk_b at 36
    table = mgr.assemble_dag_sequence("req_1", [(chunk_a, 4), (chunk_b, 36)])
    assert len(table) == 3  # 2 blocks for A, 1 block for B

    assert table.blocks[0].delta == 0   # 4 - 4
    assert table.blocks[1].delta == 0
    assert table.blocks[2].delta == 32  # 36 - 4

    # CoW Shadow block test
    old_phys_id = table.blocks[2].physical_idx
    assert not table.blocks[2].is_cow_shadow

    shadow_id = mgr.create_cow_shadow_block(table, logical_idx=2)
    assert shadow_id != old_phys_id
    assert table.blocks[2].is_cow_shadow
    assert table.blocks[2].physical_idx == shadow_id

    # Verify original canonical block pool data was copied intact
    assert torch.allclose(mgr.k_pool[shadow_id], mgr.k_pool[old_phys_id])
