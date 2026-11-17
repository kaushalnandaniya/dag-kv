"""Kernel implementations and unified dispatch for DAG-KV."""

from .dispatch import (
    adjoint_paged_attention,
    sparse_bridge_attention,
    is_triton_available,
)
from .pytorch_fallbacks import (
    pytorch_adjoint_paged_attention,
    pytorch_sparse_bridge_attention,
)

# Export legacy / direct Triton functions for backward compatibility with existing tests
try:
    from .triton_adjoint_paged_attn import (
        triton_adjoint_paged_attention,
        get_cos_sin_cache,
        _GLOBAL_COS_SIN_CACHE,
    )
    from .triton_sparse_stitch import triton_sparse_bridge_attention
except ImportError:
    triton_adjoint_paged_attention = pytorch_adjoint_paged_attention
    triton_sparse_bridge_attention = pytorch_sparse_bridge_attention
    _GLOBAL_COS_SIN_CACHE = {}

__all__ = [
    "adjoint_paged_attention",
    "sparse_bridge_attention",
    "is_triton_available",
    "pytorch_adjoint_paged_attention",
    "pytorch_sparse_bridge_attention",
    "triton_adjoint_paged_attention",
    "triton_sparse_bridge_attention",
]
