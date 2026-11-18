import tempfile
from pathlib import Path
import pytest
import torch
import torch.nn.functional as F

from dag_kv.core.sink_canonicalizer import CanonicalChunk
from dag_kv.persistence import DiskKVCache, PinnedCPUPool
from dag_kv.models import Qwen2DAGKVEngine
from transformers import Qwen2Config, Qwen2ForCausalLM


def _make_chunk(chunk_id: str, length: int = 16):
    k = torch.randn(2, 2, length, 64, dtype=torch.float32)
    v = torch.randn(2, 2, length, 64, dtype=torch.float32)
    lse = torch.zeros(2, length, dtype=torch.float32)
    token_ids = torch.arange(10, 10 + length, dtype=torch.int64)
    canonical_positions = torch.arange(4, 4 + length, dtype=torch.int64)
    return CanonicalChunk(
        chunk_id=chunk_id,
        token_ids=token_ids,
        k_cache=k,
        v_cache=v,
        lse_local=lse,
        canonical_positions=canonical_positions,
        length=length,
    )


def test_disk_kv_cache_save_and_load():
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache = DiskKVCache(cache_dir=tmp_dir)
        c1 = _make_chunk("chunk_alpha", length=24)

        assert not cache.has_chunk("chunk_alpha")
        cache.save_chunk(c1, metadata={"file": "auth.py"})
        assert cache.has_chunk("chunk_alpha")

        # Load back
        loaded = cache.load_chunk("chunk_alpha")
        assert loaded is not None
        assert loaded.chunk_id == "chunk_alpha"
        assert loaded.length == 24
        assert torch.allclose(loaded.k_cache, c1.k_cache, atol=1e-5)
        assert torch.allclose(loaded.v_cache, c1.v_cache, atol=1e-5)
        assert torch.equal(loaded.token_ids, c1.token_ids)


def test_persistence_across_simulated_restart():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # First session
        cache1 = DiskKVCache(cache_dir=tmp_dir)
        c1 = _make_chunk("repo_utils_py", length=32)
        cache1.save_chunk(c1)

        # Simulate server shutdown (cache1 discarded)
        del cache1

        # Second session (server restarts)
        cache2 = DiskKVCache(cache_dir=tmp_dir)
        assert cache2.has_chunk("repo_utils_py")
        c1_reloaded = cache2.load_chunk("repo_utils_py")
        assert c1_reloaded is not None
        assert torch.allclose(c1_reloaded.k_cache, c1.k_cache)


def test_pinned_cpu_pool_transfer():
    pool = PinnedCPUPool(
        num_layers=2,
        num_kv_heads=2,
        head_dim=64,
        max_num_blocks=16,
        block_size=16,
    )
    c = _make_chunk("chunk_pinned", length=16)
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transferred = pool.stage_and_transfer(c, device=dev)
    assert transferred.k_cache.device == dev
    assert transferred.v_cache.device == dev


def test_engine_disk_cache_zero_prefill():
    with tempfile.TemporaryDirectory() as tmp_dir:
        disk_cache = DiskKVCache(cache_dir=tmp_dir)

        cfg = Qwen2Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
        model = Qwen2ForCausalLM(cfg).eval()

        class SimpleTok:
            eos_token_id = 999
            def encode(self, t, add_special_tokens=False):
                return [max(3, ord(c) % 120) for c in t]
            def decode(self, ids, skip_special_tokens=True):
                return ""

        tok = SimpleTok()
        engine = Qwen2DAGKVEngine(
            model=model,
            tokenizer=tok,
            max_num_blocks=32,
            block_size=16,
            device="cpu",
            disk_store=disk_cache,
        )

        content = "const PORT = 8080;"
        c1 = engine.prefill_canonical_chunk(content)
        disk_cache.save_chunk(c1)
        assert disk_cache.has_chunk(c1.chunk_id)

        # Re-initialize new engine on same disk cache (simulate restart)
        engine2 = Qwen2DAGKVEngine(
            model=model,
            tokenizer=tok,
            max_num_blocks=32,
            block_size=16,
            device="cpu",
            disk_store=disk_cache,
        )
        # Prefill should hit disk cache and not invoke model forward pass!
        c2 = engine2.prefill_canonical_chunk(content)
        assert c2.chunk_id == c1.chunk_id
        assert torch.allclose(c2.k_cache.float(), c1.k_cache.float(), atol=1e-5)
