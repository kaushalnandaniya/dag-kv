"""Tests for SM75-tuned Triton Kernels on RTX 2080 Ti."""

import pytest
import torch
from dag_kv.kernels.triton_adjoint_paged_attn import (
    triton_adjoint_paged_attention,
    torch_adjoint_paged_attention_reference,
    _triton_adjoint_paged_attn_decode_kernel,
)
from dag_kv.kernels.triton_sparse_stitch import (
    triton_sparse_bridge_attention,
    torch_sparse_bridge_reference,
    _triton_sparse_bridge_micro_prefill_kernel,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU (RTX 2080 Ti)")
@pytest.mark.parametrize("num_heads,num_kv_heads", [(4, 2), (16, 2)])
@pytest.mark.parametrize("seq_len", [32, 96, 256])
def test_triton_adjoint_paged_attention_accuracy(num_heads, num_kv_heads, seq_len):
    device = torch.device("cuda:0")
    torch.manual_seed(42)

    batch_size = 2
    head_dim = 128
    block_size = 16
    max_blocks = 64
    base = 10000.0

    q = torch.randn(batch_size, num_heads, 1, head_dim, device=device, dtype=torch.float16)
    k_pool = torch.randn(max_blocks, num_kv_heads, block_size, head_dim, device=device, dtype=torch.float16)
    v_pool = torch.randn(max_blocks, num_kv_heads, block_size, head_dim, device=device, dtype=torch.float16)

    num_blocks_per_seq = (seq_len + block_size - 1) // block_size
    block_tables = torch.randint(0, max_blocks, (batch_size, num_blocks_per_seq), device=device, dtype=torch.int32)
    # Give chunks different deltas
    offset_tables = torch.randint(-50, 50, (batch_size, num_blocks_per_seq), device=device, dtype=torch.int32)
    seq_lens = torch.full((batch_size,), seq_len, device=device, dtype=torch.int32)

    # Run Triton kernel
    out_triton = triton_adjoint_paged_attention(
        q=q,
        k_pool=k_pool,
        v_pool=v_pool,
        block_tables=block_tables,
        offset_tables=offset_tables,
        seq_lens=seq_lens,
        base=base,
    )

    # Run PyTorch reference
    out_ref = torch_adjoint_paged_attention_reference(
        q=q,
        k_pool=k_pool,
        v_pool=v_pool,
        block_tables=block_tables,
        offset_tables=offset_tables,
        seq_lens=seq_lens,
        base=base,
    )

    diff = (out_triton - out_ref).abs().max().item()
    assert diff < 2e-2, f"Triton PagedAttention output diverged from reference: diff={diff}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU (RTX 2080 Ti)")
@pytest.mark.parametrize("num_bridge", [4, 16])
def test_triton_sparse_bridge_attention_accuracy(num_bridge):
    device = torch.device("cuda:0")
    torch.manual_seed(1337)

    batch_size = 1
    num_heads = 16
    num_kv_heads = 2
    head_dim = 128
    block_size = 16
    max_blocks = 32
    seq_len = 128
    base = 10000.0

    q_bridge = torch.randn(batch_size, num_heads, num_bridge, head_dim, device=device, dtype=torch.float16)
    bridge_positions = torch.linspace(10, seq_len - 1, num_bridge, device=device).to(torch.int32).unsqueeze(0)

    k_pool = torch.randn(max_blocks, num_kv_heads, block_size, head_dim, device=device, dtype=torch.float16)
    v_pool = torch.randn(max_blocks, num_kv_heads, block_size, head_dim, device=device, dtype=torch.float16)

    num_blocks_per_seq = (seq_len + block_size - 1) // block_size
    block_tables = torch.randint(0, max_blocks, (batch_size, num_blocks_per_seq), device=device, dtype=torch.int32)
    offset_tables = torch.randint(-30, 30, (batch_size, num_blocks_per_seq), device=device, dtype=torch.int32)

    # Run Triton
    out_triton = triton_sparse_bridge_attention(
        q_bridge=q_bridge,
        bridge_positions=bridge_positions,
        k_pool=k_pool,
        v_pool=v_pool,
        block_tables=block_tables,
        offset_tables=offset_tables,
        base=base,
    )

    # Run Reference
    out_ref = torch_sparse_bridge_reference(
        q_bridge=q_bridge,
        bridge_positions=bridge_positions,
        k_pool=k_pool,
        v_pool=v_pool,
        block_tables=block_tables,
        offset_tables=offset_tables,
        base=base,
    )

    diff = (out_triton - out_ref).abs().max().item()
    assert diff < 2e-2, f"Sparse bridge attention diverged: diff={diff}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU (RTX 2080 Ti)")
def test_turing_sm75_shared_memory_limits():
    """Validates that kernel shared memory is well below 64 KB (65536 bytes) on sm_75."""
    for kernel_obj, name in [
        (_triton_adjoint_paged_attn_decode_kernel, "paged_attn"),
        (_triton_sparse_bridge_micro_prefill_kernel, "sparse_bridge"),
    ]:
        assert len(kernel_obj.cache[0]) > 0, f"Kernel {name} was not compiled in previous tests"
        for key, compiled in kernel_obj.cache[0].items():
            meta = getattr(compiled, "metadata", None)
            shared = getattr(meta, "shared", getattr(compiled, "shared", 0))
            assert shared <= 65536, f"Kernel {name} exceeds 64KB shared memory: {shared} bytes"
