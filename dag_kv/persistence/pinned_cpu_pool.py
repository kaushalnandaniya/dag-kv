"""Pinned Host Memory Pool for Fast Asynchronous Staging between NVMe and GPU VRAM."""

from typing import Optional
import torch
from dag_kv.core.sink_canonicalizer import CanonicalChunk


class PinnedCPUPool:
    """Pre-allocated pinned host memory staging pool for PCIe DMA transfers."""

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_num_blocks: int = 1024,
        block_size: int = 16,
        dtype: torch.dtype = torch.float32,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_num_blocks = max_num_blocks
        self.block_size = block_size
        self.dtype = dtype

        pool_shape = (max_num_blocks, num_layers, num_kv_heads, block_size, head_dim)
        pin = torch.cuda.is_available()

        if pin:
            self.k_staging = torch.zeros(pool_shape, dtype=dtype, device="cpu").pin_memory()
            self.v_staging = torch.zeros(pool_shape, dtype=dtype, device="cpu").pin_memory()
        else:
            self.k_staging = torch.zeros(pool_shape, dtype=dtype, device="cpu")
            self.v_staging = torch.zeros(pool_shape, dtype=dtype, device="cpu")

    def stage_and_transfer(
        self,
        chunk: CanonicalChunk,
        device: torch.device,
        dtype: Optional[torch.dtype] = None,
        non_blocking: bool = True,
    ) -> CanonicalChunk:
        """Copies chunk tensors to pinned host RAM, then transfers to device asynchronously."""
        target_dtype = dtype or (torch.float16 if device.type == "cuda" else torch.float32)
        k_pinned = chunk.k_cache.to("cpu", copy=False).pin_memory() if torch.cuda.is_available() and not chunk.k_cache.is_pinned() else chunk.k_cache
        v_pinned = chunk.v_cache.to("cpu", copy=False).pin_memory() if torch.cuda.is_available() and not chunk.v_cache.is_pinned() else chunk.v_cache

        k_dev = k_pinned.to(device, dtype=target_dtype, non_blocking=non_blocking)
        v_dev = v_pinned.to(device, dtype=target_dtype, non_blocking=non_blocking)
        tok_dev = chunk.token_ids.to(device, non_blocking=non_blocking)
        lse_dev = chunk.lse_local.to(device, non_blocking=non_blocking)
        pos_dev = chunk.canonical_positions.to(device, non_blocking=non_blocking)

        return CanonicalChunk(
            chunk_id=chunk.chunk_id,
            token_ids=tok_dev,
            k_cache=k_dev,
            v_cache=v_dev,
            lse_local=lse_dev,
            canonical_positions=pos_dev,
            length=chunk.length,
            metadata=chunk.metadata,
        )
