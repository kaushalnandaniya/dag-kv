"""Paged Block Manager, Content-SHA256 Registry & CoW Shadow Blocks.

Manages physical KV cache memory on GPU:
1. Physical memory pool allocated upfront in HBM (FP16).
2. Content-addressable SHA256 storage for canonical chunks.
3. Logical-to-Physical OffsetBlockTable mapping with per-block RoPE Delta.
4. Copy-on-Write (CoW) Shadow Blocks for request-private Bridge Token updates,
   ensuring canonical blocks remain 100% read-only and shareable across requests.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Union
import torch
from .sink_canonicalizer import CanonicalChunk


@dataclass
class LogicalBlock:
    """Logical block descriptor in a sequence's block table."""
    logical_idx: int
    physical_idx: int
    delta: int                # Delta = global_start - canonical_pos_start (e.g. G - 4)
    is_cow_shadow: bool = False
    valid_tokens: int = 16


@dataclass
class PhysicalBlock:
    """Metadata for a physical block in the pool."""
    block_id: int
    ref_count: int = 0
    is_canonical: bool = False
    chunk_id: Optional[str] = None


class OffsetBlockTable:
    """Per-sequence block table maintaining physical mappings and RoPE offsets."""

    def __init__(self, seq_id: str, block_size: int = 16):
        self.seq_id = seq_id
        self.block_size = block_size
        self.blocks: List[LogicalBlock] = []

    def append_block(self, physical_idx: int, delta: int, is_cow_shadow: bool = False, valid_tokens: int = 16):
        logical_idx = len(self.blocks)
        self.blocks.append(
            LogicalBlock(
                logical_idx=logical_idx,
                physical_idx=physical_idx,
                delta=delta,
                is_cow_shadow=is_cow_shadow,
                valid_tokens=valid_tokens,
            )
        )

    def get_physical_indices(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Returns 1D tensor of physical block IDs."""
        return torch.tensor([b.physical_idx for b in self.blocks], dtype=torch.int32, device=device)

    def get_deltas(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Returns 1D tensor of integer RoPE offsets Delta."""
        return torch.tensor([b.delta for b in self.blocks], dtype=torch.int32, device=device)

    def get_valid_tokens(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Returns 1D tensor of valid token counts (1..16) per logical block."""
        return torch.tensor([b.valid_tokens for b in self.blocks], dtype=torch.int32, device=device)

    def get_block_starts(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """Returns 1D tensor of cumulative starting global positions for each logical block."""
        starts = []
        cum = 0
        for b in self.blocks:
            starts.append(cum)
            cum += b.valid_tokens
        return torch.tensor(starts, dtype=torch.int32, device=device)

    def get_chunk_groups(self) -> List[Tuple[int, List[LogicalBlock]]]:
        """Groups contiguous blocks with identical delta (representing a chunk or decode tail)."""
        if not self.blocks:
            return []
        groups = []
        cur_delta = self.blocks[0].delta
        cur_blocks = [self.blocks[0]]
        for b in self.blocks[1:]:
            if b.delta == cur_delta:
                cur_blocks.append(b)
            else:
                groups.append((cur_delta, cur_blocks))
                cur_delta = b.delta
                cur_blocks = [b]
        groups.append((cur_delta, cur_blocks))
        return groups

    def __len__(self) -> int:
        return len(self.blocks)


class BlockManager:
    """Central Paged KV Cache Memory Manager supporting Adjoint Offsets and CoW."""

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_num_blocks: int = 4096,
        block_size: int = 16,
        device: Union[str, torch.device] = "cuda:0",
        dtype: torch.dtype = torch.float16,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_num_blocks = max_num_blocks
        self.block_size = block_size
        self.device = torch.device(device)
        self.dtype = dtype

        # Enforce FP16 constraint for Turing sm_75
        assert self.dtype == torch.float16, "Hardware constraint sm_75: Memory pool must be float16"

        # Allocate physical memory pool in HBM:
        # Shape: (max_num_blocks, num_layers, num_kv_heads, block_size, head_dim)
        self.k_pool = torch.zeros(
            (max_num_blocks, num_layers, num_kv_heads, block_size, head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        self.v_pool = torch.zeros(
            (max_num_blocks, num_layers, num_kv_heads, block_size, head_dim),
            dtype=self.dtype,
            device=self.device,
        )

        # Free list & metadata tracking
        self.free_blocks: List[int] = list(range(max_num_blocks))
        self.physical_metadata: Dict[int, PhysicalBlock] = {
            i: PhysicalBlock(block_id=i) for i in range(max_num_blocks)
        }

        # Canonical Content Registry: chunk_id (SHA256) -> list of physical block IDs
        self.canonical_registry: Dict[str, List[int]] = {}
        self.canonical_lengths: Dict[str, int] = {}

    def allocate_block(self, is_canonical: bool = False, chunk_id: Optional[str] = None) -> int:
        """Allocates a single physical block."""
        if not self.free_blocks:
            raise MemoryError(f"DAG-KV BlockManager out of memory: {self.max_num_blocks} blocks exhausted")
        block_id = self.free_blocks.pop(0)
        meta = self.physical_metadata[block_id]
        meta.ref_count = 1
        meta.is_canonical = is_canonical
        meta.chunk_id = chunk_id
        return block_id

    def free_block(self, block_id: int):
        """Decrements ref_count and returns to free list if ref_count reaches 0."""
        meta = self.physical_metadata[block_id]
        meta.ref_count -= 1
        if meta.ref_count <= 0:
            meta.ref_count = 0
            meta.is_canonical = False
            meta.chunk_id = None
            self.free_blocks.append(block_id)

    def register_canonical_chunk(self, chunk: CanonicalChunk) -> List[int]:
        """Loads a CanonicalChunk into read-only physical pool blocks.

        Deduplicates via SHA256: if already cached, increments ref count.
        """
        if chunk.chunk_id in self.canonical_registry:
            block_ids = self.canonical_registry[chunk.chunk_id]
            for b_id in block_ids:
                self.physical_metadata[b_id].ref_count += 1
            return block_ids

        length = chunk.length
        num_blocks = (length + self.block_size - 1) // self.block_size
        allocated_blocks: List[int] = []

        k_cache = chunk.k_cache.to(self.device, dtype=self.dtype)
        v_cache = chunk.v_cache.to(self.device, dtype=self.dtype)

        for b_idx in range(num_blocks):
            start = b_idx * self.block_size
            end = min(start + self.block_size, length)
            block_len = end - start

            phys_id = self.allocate_block(is_canonical=True, chunk_id=chunk.chunk_id)
            allocated_blocks.append(phys_id)

            # Copy slice into physical block pool:
            # k_cache is (num_layers, num_kv_heads, L_c, head_dim)
            self.k_pool[phys_id, :, :, :block_len, :] = k_cache[:, :, start:end, :]
            self.v_pool[phys_id, :, :, :block_len, :] = v_cache[:, :, start:end, :]

        self.canonical_registry[chunk.chunk_id] = allocated_blocks
        self.canonical_lengths[chunk.chunk_id] = length
        return allocated_blocks

    def assemble_dag_sequence(
        self,
        seq_id: str,
        chunk_sequence: List[Tuple[CanonicalChunk, int]],  # List of (chunk, global_start_pos)
    ) -> OffsetBlockTable:
        """Constructs an OffsetBlockTable for a stitched sequence of chunks.

        For each chunk c starting at global position G_c:
        Its tokens sit at canonical positions p in [4, ..., L_c + 3].
        Therefore, delta = G_c - 4.
        """
        table = OffsetBlockTable(seq_id=seq_id, block_size=self.block_size)

        for chunk, global_start in chunk_sequence:
            delta = global_start - 4  # Canonical sink offset
            phys_blocks = self.register_canonical_chunk(chunk)

            remaining = chunk.length
            for p_id in phys_blocks:
                v_tokens = min(remaining, self.block_size)
                table.append_block(
                    physical_idx=p_id,
                    delta=delta,
                    is_cow_shadow=False,
                    valid_tokens=v_tokens,
                )
                remaining -= v_tokens

        return table

    def create_cow_shadow_block(self, table: OffsetBlockTable, logical_idx: int) -> int:
        """Creates a private Copy-on-Write shadow block for a bridge repair.

        Replaces the logical table entry with the new shadow block, leaving the
        original canonical physical block intact and unmodified in HBM.
        """
        log_block = table.blocks[logical_idx]
        old_phys_id = log_block.physical_idx

        # If already a private shadow block, return directly
        if log_block.is_cow_shadow:
            return old_phys_id

        # Allocate new shadow block
        new_phys_id = self.allocate_block(is_canonical=False, chunk_id=None)

        # Copy existing contents from canonical block to shadow block
        self.k_pool[new_phys_id].copy_(self.k_pool[old_phys_id])
        self.v_pool[new_phys_id].copy_(self.v_pool[old_phys_id])

        # Release refcount on old canonical block
        self.free_block(old_phys_id)

        # Update table entry
        log_block.physical_idx = new_phys_id
        log_block.is_cow_shadow = True
        return new_phys_id
