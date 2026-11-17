"""Core mathematical and structural components of DAG-KV."""

from .rope_adjoint import (
    apply_rope_half,
    apply_adjoint_rope,
    precompute_freqs_cis,
    compute_rope_angles,
    verify_adjoint_identity,
)
from .sink_canonicalizer import (
    CanonicalChunk,
    SinkCanonicalizer,
)
from .lse_probe import (
    LSEProbe,
    compute_lse_perturbation,
    select_bridge_tokens,
)
from .block_manager import (
    BlockManager,
    OffsetBlockTable,
    LogicalBlock,
    PhysicalBlock,
)
from .quantizer import (
    KVQuantizer,
)

__all__ = [
    "apply_rope_half",
    "apply_adjoint_rope",
    "precompute_freqs_cis",
    "compute_rope_angles",
    "verify_adjoint_identity",
    "CanonicalChunk",
    "SinkCanonicalizer",
    "LSEProbe",
    "compute_lse_perturbation",
    "select_bridge_tokens",
    "BlockManager",
    "OffsetBlockTable",
    "LogicalBlock",
    "PhysicalBlock",
    "KVQuantizer",
]
