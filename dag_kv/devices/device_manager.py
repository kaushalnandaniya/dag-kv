"""Hardware Abstraction Layer (HAL) & Device Manager for DAG-KV.

Discovers and abstracts hardware capabilities across NVIDIA GPUs, AMD ROCm,
Apple Silicon (MPS), and CPUs, dynamically selecting execution topologies.
"""

from dataclasses import dataclass
import os
import platform
from typing import Any, Dict, List, Optional, Tuple, Union
import torch

from .shard_config import QuantizationMode, ShardConfig, TopologyMode


@dataclass
class DeviceInfo:
    """Detailed metadata for a discovered compute device."""
    device_type: str                   # "cuda", "mps", "cpu"
    index: Optional[int]               # GPU index (0, 1, ...) or None for CPU/MPS
    name: str                          # Human-readable device name
    total_memory: int                  # Total device memory in bytes
    free_memory: int                   # Free memory in bytes
    compute_capability: Optional[Tuple[int, int]] = None  # (major, minor) for CUDA
    is_triton_supported: bool = False  # Whether Triton JIT is usable on this device

    @property
    def total_memory_gb(self) -> float:
        return self.total_memory / (1024 ** 3)

    @property
    def free_memory_gb(self) -> float:
        return self.free_memory / (1024 ** 3)


class DeviceManager:
    """Singleton Hardware Abstraction Layer manager."""

    _instance: Optional["DeviceManager"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._devices: List[DeviceInfo] = []
        self._has_triton = self._check_triton_available()
        self._refresh_devices()
        self._initialized = True

    @staticmethod
    def _check_triton_available() -> bool:
        """Checks if Triton is importable and functional."""
        try:
            import triton
            import triton.language as tl
            return True
        except ImportError:
            return False
        except Exception:
            return False

    def _refresh_devices(self):
        """Probes system and populates device list."""
        self._devices.clear()

        # 1. Probe CUDA / ROCm devices
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                free_mem, total_mem = torch.cuda.mem_get_info(i)
                cc = (props.major, props.minor)
                # Triton supports CUDA compute capability >= 7.0 (Volta, Turing sm_75, Ampere, Hopper)
                triton_ok = self._has_triton and (cc[0] >= 7)

                self._devices.append(
                    DeviceInfo(
                        device_type="cuda",
                        index=i,
                        name=props.name,
                        total_memory=total_mem,
                        free_memory=free_mem,
                        compute_capability=cc,
                        is_triton_supported=triton_ok,
                    )
                )

        # 2. Probe Apple Silicon MPS
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # MPS unified memory (estimate from system RAM)
            total_ram = self._get_system_ram()
            self._devices.append(
                DeviceInfo(
                    device_type="mps",
                    index=None,
                    name=f"Apple Silicon ({platform.processor() or 'ARM'})",
                    total_memory=total_ram,
                    free_memory=total_ram // 2,
                    compute_capability=None,
                    is_triton_supported=False,
                )
            )

        # 3. Always register CPU
        cpu_ram = self._get_system_ram()
        cpu_count = os.cpu_count() or 1
        self._devices.append(
            DeviceInfo(
                device_type="cpu",
                index=None,
                name=f"Host CPU ({cpu_count} cores, {platform.machine()})",
                total_memory=cpu_ram,
                free_memory=cpu_ram // 2,
                compute_capability=None,
                is_triton_supported=False,
            )
        )

    @staticmethod
    def _get_system_ram() -> int:
        """Returns total system host memory in bytes."""
        try:
            import psutil
            return psutil.virtual_memory().total
        except ImportError:
            try:
                # Linux /proc/meminfo fallback
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        if line.startswith("MemTotal:"):
                            return int(line.split()[1]) * 1024
            except Exception:
                pass
            return 16 * (1024 ** 3)  # 16 GB default estimate

    @property
    def devices(self) -> List[DeviceInfo]:
        return self._devices

    @property
    def has_cuda(self) -> bool:
        return any(d.device_type == "cuda" for d in self._devices)

    @property
    def has_mps(self) -> bool:
        return any(d.device_type == "mps" for d in self._devices)

    @property
    def cuda_device_count(self) -> int:
        return sum(1 for d in self._devices if d.device_type == "cuda")

    @property
    def primary_device(self) -> torch.device:
        """Returns the most capable default device."""
        if self.has_cuda:
            return torch.device("cuda:0")
        if self.has_mps:
            return torch.device("mps")
        return torch.device("cpu")

    def auto_detect_topology(
        self,
        model_size_bytes: int = 0,
        preferred_topology: Optional[str] = None,
        block_size: int = 16,
    ) -> ShardConfig:
        """Analyzes available hardware and yields the optimal ShardConfig.
        
        Args:
            model_size_bytes: Estimated model weight size in bytes.
            preferred_topology: Explicit override ("single_device", "pipeline", etc.)
            block_size: Token block size.
        """
        cuda_devices = [d for d in self._devices if d.device_type == "cuda"]

        # User explicit preference
        if preferred_topology == "pipeline" and len(cuda_devices) >= 2:
            return ShardConfig.dual_gpu_pipeline(
                target_device="cuda:0",
                prefill_device="cuda:1",
                block_size=block_size,
            )

        if preferred_topology == "cpu":
            return ShardConfig.cpu_fallback(block_size=block_size)

        # Multi-GPU topology detection
        if len(cuda_devices) >= 2:
            # 2+ CUDA GPUs available
            gpu0_free = cuda_devices[0].free_memory
            gpu1_free = cuda_devices[1].free_memory

            # If model is large or user benefits from disaggregated prefill
            if model_size_bytes > gpu0_free * 0.7:
                # Model doesn't comfortably fit on a single GPU with large KV cache
                return ShardConfig.dual_gpu_pipeline(
                    target_device="cuda:0",
                    prefill_device="cuda:1",
                    block_size=block_size,
                )
            else:
                # Fits on one, can still use disaggregated pipeline for concurrency
                return ShardConfig.dual_gpu_pipeline(
                    target_device="cuda:0",
                    prefill_device="cuda:1",
                    block_size=block_size,
                )

        # Single CUDA GPU
        if len(cuda_devices) == 1:
            dev = cuda_devices[0]
            # If VRAM is very constrained (e.g. <= 8GB) and model is large:
            if dev.total_memory_gb <= 8.5 and model_size_bytes > dev.free_memory * 0.8:
                return ShardConfig.cpu_offloaded_kv(
                    compute_device="cuda:0",
                    block_size=block_size,
                )
            return ShardConfig.single_gpu(
                device="cuda:0",
                block_size=block_size,
            )

        # Apple Silicon MPS
        if self.has_mps:
            return ShardConfig.apple_silicon_mps(block_size=block_size)

        # Fallback to CPU
        return ShardConfig.cpu_fallback(block_size=block_size)

    def allocate_kv_pool(
        self,
        max_num_blocks: int,
        num_layers: int,
        num_kv_heads: int,
        block_size: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
        pin_memory: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Allocates physical K and V pools according to device constraints."""
        shape = (max_num_blocks, num_layers, num_kv_heads, block_size, head_dim)
        
        if device.type == "cpu" and pin_memory and torch.cuda.is_available():
            k_pool = torch.zeros(shape, dtype=dtype, device="cpu").pin_memory()
            v_pool = torch.zeros(shape, dtype=dtype, device="cpu").pin_memory()
        else:
            k_pool = torch.zeros(shape, dtype=dtype, device=device)
            v_pool = torch.zeros(shape, dtype=dtype, device=device)

        return k_pool, v_pool

    def synchronize(self, device: Optional[Union[str, torch.device]] = None):
        """Cross-backend synchronization."""
        target = torch.device(device) if device is not None else self.primary_device
        if target.type == "cuda":
            torch.cuda.synchronize(target)
        elif target.type == "mps" and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()


# Global module singleton accessor
def get_device_manager() -> DeviceManager:
    return DeviceManager()
