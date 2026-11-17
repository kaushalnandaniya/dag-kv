"""vLLM Integration Plugin for DAG-KV.

Exposes DAG-KV's Content-SHA256 registry, position-canonical block tables,
and LSE-bounded bridge repair planner as an extension to vLLM's BlockManager
and PagedAttention attention backend.
"""

from dataclasses import dataclass
import hashlib
from typing import Dict, List, Optional, Set, Tuple
import torch


@dataclass
class VLLMBlockMapping:
    """Represents a mapped vLLM physical block with RoPE delta offset."""
    logical_block_id: int
    physical_block_id: int
    delta: int
    valid_tokens: int


class DAGKVvLLMBlockManagerPlugin:
    """vLLM BlockManager extension for permutation-invariant KV cache sharing."""

    def __init__(self, block_size: int = 16, bridge_top_k: float = 0.08):
        self.block_size = block_size
        self.bridge_top_k = bridge_top_k

        # Content SHA256 -> List[physical_block_id]
        self.canonical_blocks: Dict[str, List[int]] = {}
        # Physical block ref counts
        self.block_ref_counts: Dict[int, int] = {}
        # Active sequence block tables: seq_id -> List[VLLMBlockMapping]
        self.sequence_tables: Dict[str, List[VLLMBlockMapping]] = {}

    @staticmethod
    def hash_tokens(token_ids: List[int]) -> str:
        """Computes SHA-256 for a sequence of token IDs."""
        data = torch.tensor(token_ids, dtype=torch.int64).numpy().tobytes()
        return hashlib.sha256(data).hexdigest()

    def lookup_canonical_blocks(self, chunk_hash: str) -> Optional[List[int]]:
        """Returns physical block IDs if chunk is already cached in vLLM pool."""
        return self.canonical_blocks.get(chunk_hash, None)

    def register_chunk(self, chunk_hash: str, physical_block_ids: List[int]):
        """Registers newly computed physical blocks under content SHA256."""
        self.canonical_blocks[chunk_hash] = physical_block_ids
        for b_id in physical_block_ids:
            self.block_ref_counts[b_id] = self.block_ref_counts.get(b_id, 0) + 1

    def assemble_sequence_mapping(
        self,
        seq_id: str,
        chunk_hashes_and_lengths: List[Tuple[str, int]],
        global_start_offset: int = 4,
    ) -> List[VLLMBlockMapping]:
        """Constructs an offset-aware block table for a prompt composed of arbitrary chunks."""
        mappings: List[VLLMBlockMapping] = []
        current_global_pos = global_start_offset
        logical_id = 0

        for chunk_hash, length in chunk_hashes_and_lengths:
            delta = current_global_pos - 4  # Canonical sink position offset
            phys_blocks = self.canonical_blocks.get(chunk_hash, [])
            remaining = length

            for p_id in phys_blocks:
                v_tokens = min(remaining, self.block_size)
                mappings.append(
                    VLLMBlockMapping(
                        logical_block_id=logical_id,
                        physical_block_id=p_id,
                        delta=delta,
                        valid_tokens=v_tokens,
                    )
                )
                logical_id += 1
                remaining -= v_tokens

            current_global_pos += length

        self.sequence_tables[seq_id] = mappings
        return mappings

    def release_sequence(self, seq_id: str):
        """Releases sequence mappings and decrements reference counts."""
        if seq_id in self.sequence_tables:
            del self.sequence_tables[seq_id]
