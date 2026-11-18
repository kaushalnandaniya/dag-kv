import asyncio
import json
import socket
import urllib.request
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from dag_kv.models import Qwen2DAGKVEngine
from dag_kv.serving import AsyncServer, ContinuousBatchingScheduler


class _DummyTokenizer:
    eos_token_id = 9999

    def encode(self, text, add_special_tokens=False):
        return [max(3, ord(c) % 250) for c in text]

    def decode(self, ids, skip_special_tokens=True):
        if isinstance(ids, int):
            ids = [ids]
        return "".join(chr(65 + (i % 26)) for i in ids)


def _make_engine():
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev.startswith("cuda") else torch.float32

    cfg = Qwen2Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=1024,
    )
    model = Qwen2ForCausalLM(cfg).to(device=dev, dtype=dtype).eval()
    tok = _DummyTokenizer()

    return Qwen2DAGKVEngine(
        model=model,
        tokenizer=tok,
        max_num_blocks=64,
        block_size=16,
        device=dev,
        dtype=dtype,
    )


def test_continuous_batching_scheduler():
    async def _run():
        engine = _make_engine()
        bm = engine.block_manager
        free_init = len(bm.free_blocks)

        scheduler = ContinuousBatchingScheduler(engine=engine, max_batch_size=4, step_interval_s=0.001)
        scheduler.start()

        try:
            # Launch 4 concurrent requests with different file permutations
            files_a = ["### FILE: a.py\ndef fa(): return 1\n", "### FILE: b.py\ndef fb(): return 2\n"]
            files_b = ["### FILE: b.py\ndef fb(): return 2\n", "### FILE: a.py\ndef fa(): return 1\n"]

            async def _client_req(files, q_suffix):
                content = "\n".join(files) + f"\nQuestion: {q_suffix}?"
                resp = await scheduler.generate_response(
                    messages=[{"role": "user", "content": content}],
                    max_tokens=6,
                )
                return resp

            results = await asyncio.gather(
                _client_req(files_a, "QA"),
                _client_req(files_b, "QB"),
                _client_req(files_a, "QC"),
                _client_req(files_b, "QD"),
            )

            assert len(results) == 4
            for res in results:
                assert res["object"] == "chat.completion"
                assert len(res["choices"][0]["message"]["content"]) > 0
                assert res["usage"]["completion_tokens"] == 6

            # After all requests finish, all sequence blocks must be returned to free list
            sink_blocks = 1 if engine._sink_phys_block is not None else 0
            assert len(bm.free_blocks) == free_init - (sum(len(v) for v in bm.canonical_registry.values()) + sink_blocks)

        finally:
            scheduler.stop()

    asyncio.run(_run())


def test_async_server_http_and_sse():
    async def _run():
        engine = _make_engine()
        scheduler = ContinuousBatchingScheduler(engine=engine, max_batch_size=4, step_interval_s=0.001)

        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()

        server = AsyncServer(scheduler=scheduler, host="127.0.0.1", port=port)
        await server.start()

        try:
            base_url = f"http://127.0.0.1:{port}"

            # 1. Health check
            def _get_health():
                with urllib.request.urlopen(f"{base_url}/health", timeout=10) as resp:
                    assert resp.status == 200
                    data = json.loads(resp.read().decode("utf-8"))
                    assert data["status"] == "ok"
            await asyncio.to_thread(_get_health)

            # 2. Pool stats
            def _get_stats():
                with urllib.request.urlopen(f"{base_url}/v1/pool/stats", timeout=10) as resp:
                    assert resp.status == 200
                    stats = json.loads(resp.read().decode("utf-8"))
                    assert "free_blocks" in stats
            await asyncio.to_thread(_get_stats)

            # 3. Non-streaming completion
            content = "### FILE: test.py\nx = 42\n\nQuestion: What is x?"
            payload = json.dumps({
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 4,
                "stream": False,
            }).encode("utf-8")

            def _post_non_stream():
                req = urllib.request.Request(
                    f"{base_url}/v1/chat/completions",
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=15) as resp:
                    assert resp.status == 200
                    ans = json.loads(resp.read().decode("utf-8"))
                    assert ans["object"] == "chat.completion"
                    assert len(ans["choices"]) == 1
            await asyncio.to_thread(_post_non_stream)

            # 4. SSE streaming completion
            stream_payload = json.dumps({
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 4,
                "stream": True,
            }).encode("utf-8")

            def _post_stream():
                req_stream = urllib.request.Request(
                    f"{base_url}/v1/chat/completions",
                    data=stream_payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req_stream, timeout=15) as resp:
                    assert resp.status == 200
                    raw_sse = resp.read().decode("utf-8")
                    assert "data: " in raw_sse
                    assert "data: [DONE]" in raw_sse
            await asyncio.to_thread(_post_stream)

        finally:
            await server.stop()

    asyncio.run(_run())
