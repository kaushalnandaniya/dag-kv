"""Unit tests for Query-Side Adjoint RoPE Mathematical Identity."""

import pytest
import torch
from dag_kv.core.rope_adjoint import (
    rotate_half,
    compute_inv_freq,
    compute_rope_angles,
    apply_rope_half,
    apply_adjoint_rope,
    verify_adjoint_identity,
)


@pytest.mark.parametrize("dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("base", [10000.0, 1000000.0])
def test_rope_adjoint_identity_scalar(dim, dtype, base):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # Test cases: (n, p, delta)
    test_cases = [
        (100, 10, 50),     # target pos = 60
        (512, 4, 128),     # target pos = 132
        (2048, 20, 1024),  # target pos = 1044
        (150, 20, -10),    # negative delta: target pos = 10
        (4, 4, 0),         # zero delta
    ]

    atol = 1e-4 if dtype == torch.float32 else 5e-2

    for n, p, delta in test_cases:
        q = torch.randn(dim, device=device, dtype=dtype)
        k = torch.randn(dim, device=device, dtype=dtype)

        passed, s_glob, s_adj, err = verify_adjoint_identity(
            q, k, n, p, delta, dim=dim, base=base, atol=atol
        )
        assert passed, f"Failed at n={n}, p={p}, delta={delta}, err={err:.6f}"


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("num_heads", [2, 16])
@pytest.mark.parametrize("head_dim", [128])
def test_rope_adjoint_batched_multihead(batch_size, num_heads, head_dim):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1337)

    n = 256
    p = 16
    delta = 64
    base = 10000.0

    q = torch.randn(batch_size, num_heads, 1, head_dim, device=device, dtype=torch.float32)
    k = torch.randn(batch_size, num_heads, 1, head_dim, device=device, dtype=torch.float32)

    cos_n, sin_n = compute_rope_angles(n, head_dim, base=base, device=device)
    cos_target, sin_target = compute_rope_angles(p + delta, head_dim, base=base, device=device)
    cos_p, sin_p = compute_rope_angles(p, head_dim, base=base, device=device)

    # Global rotated
    q_glob = apply_rope_half(q, cos_n, sin_n, unsqueeze_dim=1)
    k_glob = apply_rope_half(k, cos_target, sin_target, unsqueeze_dim=1)
    score_glob = (q_glob * k_glob).sum(dim=-1)

    # Adjoint rotated
    k_can = apply_rope_half(k, cos_p, sin_p, unsqueeze_dim=1)
    q_adj = apply_adjoint_rope(q_glob, delta, dim=head_dim, base=base)
    score_adj = (q_adj * k_can).sum(dim=-1)

    assert torch.allclose(score_glob, score_adj, atol=1e-4), (
        f"Multihead adjoint failed: max err = {(score_glob - score_adj).abs().max().item()}"
    )


def test_rotate_half_orthogonality():
    """Verify that rotate_half is an isometry and R^4 = I."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    x = torch.randn(8, 128, device=device)
    r1 = rotate_half(x)
    r2 = rotate_half(r1)
    r3 = rotate_half(r2)
    r4 = rotate_half(r3)

    # R^2 = -I
    assert torch.allclose(r2, -x, atol=1e-6)
    # R^4 = I
    assert torch.allclose(r4, x, atol=1e-6)
    # Norm is preserved: ||R(x)|| == ||x||
    assert torch.allclose(torch.norm(x, dim=-1), torch.norm(r1, dim=-1), atol=1e-6)
