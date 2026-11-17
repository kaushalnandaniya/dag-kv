"""Tests for Hugging Face and vLLM Integrations."""

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from dag_kv.integrations.huggingface import DAGKVModelWrapper, patch_hf_model
from dag_kv.integrations.vllm import DAGKVvLLMBlockManagerPlugin
from dag_kv.devices.shard_config import ShardConfig


class DummyTokenizer:
    """Mock tokenizer for offline testing."""
    def __init__(self):
        self.vocab_size = 100
        self.eos_token_id = 99

    def encode(self, text, add_special_tokens=False):
        # Deterministic dummy tokenization based on characters
        return [min(ord(c) % 80 + 10, self.vocab_size - 1) for c in text.strip()]

    def decode(self, token_ids, skip_special_tokens=True):
        return "".join([chr((t - 10) % 80 + 32) if t >= 10 else "?" for t in token_ids])


def test_vllm_block_manager_plugin():
    """Validates vLLM plugin block mapping and permutation reuse."""
    plugin = DAGKVvLLMBlockManagerPlugin(block_size=16)

    # Hash and register chunk A (32 tokens -> 2 blocks)
    tokens_a = list(range(32))
    hash_a = plugin.hash_tokens(tokens_a)
    plugin.register_chunk(hash_a, [101, 102])

    # Hash and register chunk B (16 tokens -> 1 block)
    tokens_b = list(range(32, 48))
    hash_b = plugin.hash_tokens(tokens_b)
    plugin.register_chunk(hash_b, [201])

    # Lookup
    assert plugin.lookup_canonical_blocks(hash_a) == [101, 102]
    assert plugin.lookup_canonical_blocks(hash_b) == [201]

    # Assemble sequence [Chunk_B, Chunk_A] (permutation reuse)
    mappings = plugin.assemble_sequence_mapping(
        seq_id="req_1",
        chunk_hashes_and_lengths=[(hash_b, 16), (hash_a, 32)],
        global_start_offset=4,
    )

    assert len(mappings) == 3
    # Chunk B is at global position 4 -> delta = 4 - 4 = 0
    assert mappings[0].physical_block_id == 201
    assert mappings[0].delta == 0
    # Chunk A is at global position 4 + 16 = 20 -> delta = 20 - 4 = 16
    assert mappings[1].physical_block_id == 101
    assert mappings[1].delta == 16
    assert mappings[2].physical_block_id == 102
    assert mappings[2].delta == 16


def test_hf_wrapper_cpu_offline():
    """Tests DAGKVModelWrapper on a dummy Qwen2 model executing entirely on CPU."""
    vocab_size = 256
    config = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=512,
    )
    model = Qwen2ForCausalLM(config).eval()
    tokenizer = DummyTokenizer()
    tokenizer.vocab_size = vocab_size

    shard_cfg = ShardConfig.cpu_fallback(block_size=8, max_num_blocks=32)
    wrapper = patch_hf_model(model, tokenizer, shard_config=shard_cfg)

    chunk_1 = wrapper.prefill_chunk("Hello world from DAG-KV")
    assert chunk_1.length > 0
    assert chunk_1.k_cache.shape[0] == 2  # 2 layers

    chunk_2 = wrapper.prefill_chunk("Testing universal multi-device execution")
    assert chunk_2.length > 0

    # Test generation
    completion = wrapper.generate([chunk_1, chunk_2], prompt_suffix="Output:", max_new_tokens=4)
    assert isinstance(completion, str)
