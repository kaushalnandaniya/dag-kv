import pytest
import torch
from dag_kv.core.block_manager import BlockManager
from dag_kv.core.sink_canonicalizer import CanonicalChunk


def _make_dummy_chunk(chunk_id: str, length: int = 16, num_layers: int = 2, num_heads: int = 2, head_dim: int = 64):
    k = torch.randn(num_layers, num_heads, length, head_dim, dtype=torch.float32)
    v = torch.randn(num_layers, num_heads, length, head_dim, dtype=torch.float32)
    lse = torch.zeros(num_heads, length, dtype=torch.float32)
    token_ids = torch.arange(10, 10 + length, dtype=torch.int64)
    canonical_positions = torch.arange(4, 4 + length, dtype=torch.int64)
    return CanonicalChunk(
        chunk_id=chunk_id,
        token_ids=token_ids,
        k_cache=k,
        v_cache=v,
        lse_local=lse,
        canonical_positions=canonical_positions,
        length=length,
    )


def test_lru_eviction_lifecycle():
    # Pool with exactly 4 blocks, block_size = 16
    bm = BlockManager(
        num_layers=2,
        num_kv_heads=2,
        head_dim=64,
        max_num_blocks=4,
        block_size=16,
        device="cpu",
        enable_lru=True,
    )

    c1 = _make_dummy_chunk("chunk_1", length=16)  # takes 1 block
    c2 = _make_dummy_chunk("chunk_2", length=16)  # takes 1 block
    c3 = _make_dummy_chunk("chunk_3", length=16)  # takes 1 block
    c4 = _make_dummy_chunk("chunk_4", length=16)  # takes 1 block

    b1 = bm.register_canonical_chunk(c1)
    b2 = bm.register_canonical_chunk(c2)
    b3 = bm.register_canonical_chunk(c3)
    b4 = bm.register_canonical_chunk(c4)

    assert len(bm.free_blocks) == 0
    assert bm.evictions == 0

    # Touch c1 so c2 is now the oldest
    bm.touch("chunk_1")

    # Registering c5 should trigger automatic LRU eviction of c2!
    c5 = _make_dummy_chunk("chunk_5", length=16)
    b5 = bm.register_canonical_chunk(c5)

    assert bm.evictions == 1
    assert "chunk_2" not in bm.canonical_registry
    assert "chunk_1" in bm.canonical_registry
    assert "chunk_5" in bm.canonical_registry
    assert len(bm.free_blocks) == 0


def test_lru_protection_of_active_sequences():
    # Pool with 3 blocks
    bm = BlockManager(
        num_layers=2,
        num_kv_heads=2,
        head_dim=64,
        max_num_blocks=3,
        block_size=16,
        device="cpu",
        enable_lru=True,
    )

    c1 = _make_dummy_chunk("active_chunk", length=16)
    c2 = _make_dummy_chunk("cold_chunk", length=16)

    bm.register_canonical_chunk(c1)
    bm.register_canonical_chunk(c2)

    # Assemble sequence using c1 -> ref_count becomes 2
    table = bm.assemble_dag_sequence("seq_1", [(c1, 4)])
    p_id = table.blocks[0].physical_idx
    assert bm.physical_metadata[p_id].ref_count == 2
    assert not bm.is_chunk_evictable("active_chunk")
    assert bm.is_chunk_evictable("cold_chunk")

    # Allocate remaining block
    c3 = _make_dummy_chunk("chunk_3", length=16)
    bm.register_canonical_chunk(c3)
    assert len(bm.free_blocks) == 0

    # Allocating another chunk must evict cold_chunk or chunk_3, NOT active_chunk!
    c4 = _make_dummy_chunk("chunk_4", length=16)
    bm.register_canonical_chunk(c4)

    assert "active_chunk" in bm.canonical_registry
    assert bm.physical_metadata[p_id].ref_count == 2
    assert bm.evictions == 1

    # Free sequence -> ref_count drops back to 1
    bm.free_sequence(table)
    assert bm.physical_metadata[p_id].ref_count == 1
    assert bm.is_chunk_evictable("active_chunk")


def test_ensure_capacity_and_exhaustion():
    bm = BlockManager(
        num_layers=2,
        num_kv_heads=2,
        head_dim=64,
        max_num_blocks=3,
        block_size=16,
        device="cpu",
        enable_lru=True,
    )
    c1 = _make_dummy_chunk("c1", length=16)
    c2 = _make_dummy_chunk("c2", length=16)
    c3 = _make_dummy_chunk("c3", length=16)
    bm.register_canonical_chunk(c1)
    bm.register_canonical_chunk(c2)
    bm.register_canonical_chunk(c3)
    assert len(bm.free_blocks) == 0

    # Ensure 2 blocks free
    freed = bm.ensure_capacity(blocks_needed=2, protected_chunks={"c3"})
    assert freed == 2
    assert len(bm.free_blocks) == 2
    assert "c3" in bm.canonical_registry
    assert "c1" not in bm.canonical_registry
    assert "c2" not in bm.canonical_registry

    # Active sequence pinned on c3
    table = bm.assemble_dag_sequence("seq_pinned", [(c3, 4)])
    # Consume remaining free blocks
    bm.allocate_block()
    bm.allocate_block()
    assert len(bm.free_blocks) == 0

    # All blocks in use and active sequence pinned -> allocation must raise MemoryError
    with pytest.raises(MemoryError):
        bm.allocate_block()

    bm.free_sequence(table)
