"""Tests for Pure PyTorch Fallback Kernels and Backend Parity."""

import math
import pytest
import torch

from dag_kv.core.rope_adjoint import compute_rope_angles
from dag_kv.kernels.pytorch_fallbacks import (
    pytorch_adjoint_paged_attention,
    pytorch_sparse_bridge_attention,
)
from dag_kv.kernels.dispatch import (
    adjoint_paged_attention,
    sparse_bridge_attention,
    is_triton_available,
)

if is_triton_available():
    from dag_kv.kernels.triton_adjoint_paged_attn import triton_adjoint_paged_attention
    from dag_kv.kernels.triton_sparse_stitch import triton_sparse_bridge_attention


def test_pytorch_adjoint_paged_attention_cpu():
    """Validates that PyTorch fallback executes cleanly on CPU."""
    batch_size = 2
    num_heads = 4
    num_kv_heads = 2
    head_dim = 64
    block_size = 16
    max_blocks = 8
    seq_len = 32

    q = torch.randn(batch_size, num_heads, 1, head_dim, dtype=torch.float32, device="cpu")
    k_pool = torch.randn(max_blocks, num_kv_heads, block_size, head_dim, dtype=torch.float32, device="cpu")
    v_pool = torch.randn(max_blocks, num_kv_heads, block_size, head_dim, dtype=torch.float32, device="cpu")

    block_tables = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32, device="cpu")
    offset_tables = torch.tensor([[0, 16], [5, 21]], dtype=torch.int32, device="cpu")
    seq_lens = torch.tensor([seq_len, seq_len], dtype=torch.int32, device="cpu")
    block_starts = torch.tensor([[0, 16], [0, 16]], dtype=torch.int32, device="cpu")
    valid_lens = torch.tensor([[16, 16], [16, 16]], dtype=torch.int32, device="cpu")

    out = pytorch_adjoint_paged_attention(
        q=q,
        k_pool=k_pool,
        v_pool=v_pool,
        block_tables=block_tables,
        offset_tables=offset_tables,
        seq_lens=seq_lens,
        block_start_tables=block_starts,
        valid_lens_tables=valid_lens,
    )

    assert out.shape == (batch_size, num_heads, 1, head_dim)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


@pytest.mark.skipif(not torch.cuda.is_available() or not is_triton_available(), reason="CUDA + Triton required")
def test_pytorch_vs_triton_adjoint_paged_attention_parity():
    """Verifies that PyTorch fallback matches Triton kernel numerically within FP16 tolerance."""
    device = "cuda:0"
    batch_size = 1
    num_heads = 4
    num_kv_heads = 2
    head_dim = 64
    block_size = 16
    max_blocks = 8
    seq_len = 32

    torch.manual_seed(42)
    q = torch.randn(batch_size, num_heads, 1, head_dim, dtype=torch.float16, device=device)
    k_pool = torch.randn(max_blocks, num_kv_heads, block_size, head_dim, dtype=torch.float16, device=device)
    v_pool = torch.randn(max_blocks, num_kv_heads, block_size, head_dim, dtype=torch.float16, device=device)

    block_tables = torch.tensor([[0, 1]], dtype=torch.int32, device=device)
    offset_tables = torch.tensor([[0, 16]], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)
    block_starts = torch.tensor([[0, 16]], dtype=torch.int32, device=device)
    valid_lens = torch.tensor([[16, 16]], dtype=torch.int32, device=device)

    out_triton = triton_adjoint_paged_attention(
        q=q,
        k_pool=k_pool,
        v_pool=v_pool,
        block_tables=block_tables,
        offset_tables=offset_tables,
        seq_lens=seq_lens,
        block_start_tables=block_starts,
        valid_lens_tables=valid_lens,
    )

    out_pytorch = pytorch_adjoint_paged_attention(
        q=q,
        k_pool=k_pool,
        v_pool=v_pool,
        block_tables=block_tables,
        offset_tables=offset_tables,
        seq_lens=seq_lens,
        block_start_tables=block_starts,
        valid_lens_tables=valid_lens,
    )

    diff = torch.abs(out_triton.float() - out_pytorch.float()).max().item()
    # FP16 accumulation tolerance
    assert diff < 5e-2, f"Max diff between Triton and PyTorch fallback: {diff}"


@pytest.mark.skipif(not torch.cuda.is_available() or not is_triton_available(), reason="CUDA + Triton required")
def test_pytorch_vs_triton_sparse_bridge_parity():
    """Verifies that PyTorch fallback matches Triton sparse stitch kernel."""
    device = "cuda:0"
    batch_size = 1
    num_heads = 4
    num_kv_heads = 2
    num_bridge = 2
    head_dim = 64
    block_size = 16
    max_blocks = 8

    torch.manual_seed(42)
    q_bridge = torch.randn(batch_size, num_heads, num_bridge, head_dim, dtype=torch.float16, device=device)
    bridge_pos = torch.tensor([[10, 25]], dtype=torch.int32, device=device)
    k_pool = torch.randn(max_blocks, num_kv_heads, block_size, head_dim, dtype=torch.float16, device=device)
    v_pool = torch.randn(max_blocks, num_kv_heads, block_size, head_dim, dtype=torch.float16, device=device)

    block_tables = torch.tensor([[0, 1]], dtype=torch.int32, device=device)
    offset_tables = torch.tensor([[0, 16]], dtype=torch.int32, device=device)
    block_starts = torch.tensor([[0, 16]], dtype=torch.int32, device=device)
    valid_lens = torch.tensor([[16, 16]], dtype=torch.int32, device=device)

    out_triton = triton_sparse_bridge_attention(
        q_bridge=q_bridge,
        bridge_positions=bridge_pos,
        k_pool=k_pool,
        v_pool=v_pool,
        block_tables=block_tables,
        offset_tables=offset_tables,
        block_start_tables=block_starts,
        valid_lens_tables=valid_lens,
    )

    out_pytorch = pytorch_sparse_bridge_attention(
        q_bridge=q_bridge,
        bridge_positions=bridge_pos,
        k_pool=k_pool,
        v_pool=v_pool,
        block_tables=block_tables,
        offset_tables=offset_tables,
        block_start_tables=block_starts,
        valid_lens_tables=valid_lens,
    )

    diff = torch.abs(out_triton.float() - out_pytorch.float()).max().item()
    assert diff < 5e-2, f"Max diff between Triton and PyTorch sparse stitch: {diff}"
