"""Pluggable Shard Configuration and Topology Specifications for DAG-KV.

Supports Single-Device, Multi-GPU Pipeline, Tensor Parallelism,
and CPU-Offloaded KV Pooling across NVIDIA, AMD ROCm, Apple Silicon (MPS), and CPU.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union
import torch


class TopologyMode(str, Enum):
    """Topology execution modes supported by DAG-KV."""
    SINGLE_DEVICE = "single_device"    # All layers and KV cache on one device
    PIPELINE = "pipeline"              # Prefill and decode disaggregated across devices
    TENSOR_PARALLEL = "tensor_parallel" # Distributed execution
    CPU_OFFLOAD = "cpu_offload"        # Weights on accelerator, KV cache pool in Host RAM


class QuantizationMode(str, Enum):
    """KV cache quantization precision."""
    NONE = "none"                      # Native precision (FP16, BF16, or FP32)
    FP16 = "fp16"                      # 16-bit floating point
    INT8 = "int8"                      # 8-bit symmetric quantization (50% VRAM reduction)
    INT4 = "int4"                      # 4-bit packed quantization (75% VRAM reduction)


@dataclass
class ShardConfig:
    """Pluggable hardware topology and memory allocation configuration."""
    
    # Primary compute device for inference decode (e.g. "cuda:0", "mps", "cpu")
    device: Union[str, torch.device] = "cuda:0"
    
    # Device where the KV cache pool resides (can be "cpu" for Host RAM offloading)
    kv_device: Optional[Union[str, torch.device]] = None
    
    # Execution topology mode
    topology: TopologyMode = TopologyMode.SINGLE_DEVICE
    
    # List of all devices participating in execution
    devices: List[Union[str, torch.device]] = field(default_factory=lambda: ["cuda:0"])
    
    # Dedicated prefill device for disaggregated pipeline mode (e.g. "cuda:1")
    prefill_device: Optional[Union[str, torch.device]] = None
    
    # Explicit mapping of model layer index -> target device (e.g. {0: "cuda:0", 14: "cuda:1"})
    layer_distribution: Optional[Dict[int, str]] = None
    
    # Memory pool data type
    dtype: torch.dtype = torch.float16
    
    # KV cache quantization mode
    quantization: QuantizationMode = QuantizationMode.NONE
    
    # Dynamic block size for PagedAttention (tokens per physical block)
    block_size: int = 16
    
    # Maximum physical blocks to pre-allocate in memory pool
    max_num_blocks: int = 2048
    
    # Whether to use pinned (page-locked) host memory for DMA transfers and CPU offload
    pin_memory: bool = True
    
    # LSE bridge selection top-k percentage (default 8% for theoretical bound)
    bridge_top_k: float = 0.08

    def __post_init__(self):
        self.device = torch.device(self.device) if isinstance(self.device, str) else self.device
        if self.kv_device is None:
            self.kv_device = self.device
        else:
            self.kv_device = torch.device(self.kv_device) if isinstance(self.kv_device, str) else self.kv_device
            
        if self.prefill_device is not None:
            self.prefill_device = torch.device(self.prefill_device) if isinstance(self.prefill_device, str) else self.prefill_device

        # Normalize devices list
        self.devices = [torch.device(d) if isinstance(d, str) else d for d in self.devices]

        # Ensure dtype is appropriate for device
        if self.device.type == "cpu" and self.dtype == torch.float16:
            # CPU PyTorch has limited FP16 math support on older CPUs; float32 or bfloat16 is safer
            self.dtype = torch.float32

    @property
    def is_cuda(self) -> bool:
        return self.device.type == "cuda"

    @property
    def is_mps(self) -> bool:
        return self.device.type == "mps"

    @property
    def is_cpu(self) -> bool:
        return self.device.type == "cpu"

    @property
    def is_cpu_offloaded(self) -> bool:
        return self.kv_device.type == "cpu" and self.device.type != "cpu"

    @classmethod
    def single_gpu(
        cls,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.float16,
        block_size: int = 16,
        max_num_blocks: int = 2048,
        quantization: QuantizationMode = QuantizationMode.NONE,
    ) -> "ShardConfig":
        """Convenience constructor for standard single-GPU environments."""
        return cls(
            device=device,
            kv_device=device,
            topology=TopologyMode.SINGLE_DEVICE,
            devices=[device],
            dtype=dtype,
            block_size=block_size,
            max_num_blocks=max_num_blocks,
            quantization=quantization,
        )

    @classmethod
    def dual_gpu_pipeline(
        cls,
        target_device: str = "cuda:0",
        prefill_device: str = "cuda:1",
        dtype: torch.dtype = torch.float16,
        block_size: int = 16,
        max_num_blocks: int = 2048,
    ) -> "ShardConfig":
        """Convenience constructor for dual-GPU disaggregated pipeline."""
        return cls(
            device=target_device,
            kv_device=target_device,
            prefill_device=prefill_device,
            topology=TopologyMode.PIPELINE,
            devices=[target_device, prefill_device],
            dtype=dtype,
            block_size=block_size,
            max_num_blocks=max_num_blocks,
        )

    @classmethod
    def cpu_fallback(
        cls,
        block_size: int = 16,
        max_num_blocks: int = 1024,
        dtype: torch.dtype = torch.float32,
    ) -> "ShardConfig":
        """Convenience constructor for pure CPU fallback environments."""
        return cls(
            device="cpu",
            kv_device="cpu",
            topology=TopologyMode.SINGLE_DEVICE,
            devices=["cpu"],
            dtype=dtype,
            block_size=block_size,
            max_num_blocks=max_num_blocks,
        )

    @classmethod
    def apple_silicon_mps(
        cls,
        block_size: int = 16,
        max_num_blocks: int = 1024,
        dtype: torch.dtype = torch.float32,
    ) -> "ShardConfig":
        """Convenience constructor for Apple Silicon (MPS)."""
        return cls(
            device="mps",
            kv_device="mps",
            topology=TopologyMode.SINGLE_DEVICE,
            devices=["mps"],
            dtype=dtype,
            block_size=block_size,
            max_num_blocks=max_num_blocks,
        )

    @classmethod
    def cpu_offloaded_kv(
        cls,
        compute_device: str = "cuda:0",
        max_num_blocks: int = 8192,
        block_size: int = 16,
        dtype: torch.dtype = torch.float16,
    ) -> "ShardConfig":
        """Convenience constructor for systems with small GPU VRAM and large Host RAM."""
        return cls(
            device=compute_device,
            kv_device="cpu",
            topology=TopologyMode.CPU_OFFLOAD,
            devices=[compute_device, "cpu"],
            dtype=dtype,
            block_size=block_size,
            max_num_blocks=max_num_blocks,
            pin_memory=True,
        )
