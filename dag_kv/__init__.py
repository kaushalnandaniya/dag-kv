"""DAG-KV: Position-Decoupled Graph KV Cache with Query-Side Adjoint RoPE & LSE-Bounded Sparse Bridge Repair."""

__version__ = "0.3.4"

from .core.block_manager import BlockManager, OffsetBlockTable, LogicalBlock, PhysicalBlock
from .core.lse_probe import LSEProbe, compute_lse_perturbation, select_bridge_tokens
from .core.quantizer import KVQuantizer
from .core.rope_adjoint import apply_adjoint_rope, apply_rope_half, compute_rope_angles
from .core.sink_canonicalizer import CanonicalChunk, SinkCanonicalizer
from .devices.device_manager import DeviceInfo, DeviceManager, get_device_manager
from .devices.shard_config import QuantizationMode, ShardConfig, TopologyMode
from .integrations.huggingface import DAGKVModelWrapper, patch_hf_model
from .integrations.vllm import DAGKVvLLMBlockManagerPlugin
from .kernels.dispatch import adjoint_paged_attention, is_triton_available, sparse_bridge_attention
from .models.base_engine import DAGKVEngine
from .models.auto_engine import AutoDAGKVEngine
from .models.llama_engine import LlamaDAGKVEngine
from .models.mistral_engine import MistralDAGKVEngine
from .models.gemma_engine import Gemma2DAGKVEngine
from .models.qwen2_dag_patch import Qwen2DAGKVEngine
from .persistence.disk_cache import DiskKVCache
from .persistence.pinned_cpu_pool import PinnedCPUPool
from .serving.async_server import AsyncServer
from .serving.scheduler import ContinuousBatchingScheduler
from .worker.dual_gpu_pipeline import DualGPUPipeline
from .worker.universal_pipeline import UniversalPipeline

__all__ = [
    "__version__",
    "BlockManager",
    "OffsetBlockTable",
    "LogicalBlock",
    "PhysicalBlock",
    "LSEProbe",
    "compute_lse_perturbation",
    "select_bridge_tokens",
    "KVQuantizer",
    "apply_adjoint_rope",
    "apply_rope_half",
    "compute_rope_angles",
    "CanonicalChunk",
    "SinkCanonicalizer",
    "DeviceInfo",
    "DeviceManager",
    "get_device_manager",
    "QuantizationMode",
    "ShardConfig",
    "TopologyMode",
    "DAGKVModelWrapper",
    "patch_hf_model",
    "DAGKVvLLMBlockManagerPlugin",
    "adjoint_paged_attention",
    "sparse_bridge_attention",
    "is_triton_available",
    "DAGKVEngine",
    "AutoDAGKVEngine",
    "Qwen2DAGKVEngine",
    "LlamaDAGKVEngine",
    "MistralDAGKVEngine",
    "Gemma2DAGKVEngine",
    "DiskKVCache",
    "PinnedCPUPool",
    "ContinuousBatchingScheduler",
    "AsyncServer",
    "DualGPUPipeline",
    "UniversalPipeline",
]
