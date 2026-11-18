import hashlib
import json
import math
import re
import socket
import threading
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
import torch.nn.functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM

from dag_kv import (
    DAGKVModelWrapper,
    KVQuantizer,
    adjoint_paged_attention,
    compute_lse_perturbation,
    select_bridge_tokens,
)


class _StressTokenizer:
    eos_token_id = 999999

    def encode(self, text, add_special_tokens=False):
        return [max(3, (ord(c) % 500)) for c in text]

    def decode(self, ids, skip_special_tokens=True):
        if isinstance(ids, int):
            ids = [ids]
        return "".join(chr(65 + (i % 26)) for i in ids)


def _make_tiny_qwen(device: str, dtype: torch.dtype) -> Qwen2ForCausalLM:
    cfg = Qwen2Config(
        vocab_size=512,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=2048,
    )
    return Qwen2ForCausalLM(cfg).to(device=device, dtype=dtype).eval()


def test_stress_1_sha256_dedup_and_permutation_invariance():
    torch.manual_seed(42)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev.startswith("cuda") else torch.float32

    wrapper = DAGKVModelWrapper(
        model=_make_tiny_qwen(dev, dtype),
        tokenizer=_StressTokenizer(),
        max_num_blocks=64,
        block_size=16,
        quantization="none",
        bridge_top_k=0.2,
    )
    bm = wrapper.engine.block_manager

    c1 = wrapper.prefill_chunk("Canonical file segment alpha " * 2)
    c1_dup = wrapper.prefill_chunk("Canonical file segment alpha " * 2)
    c2 = wrapper.prefill_chunk("Canonical file segment beta " * 2)

    assert c1.chunk_id == c1_dup.chunk_id
    free_after_prefill = len(bm.free_blocks)

    out_ab = wrapper.generate([c1, c2], prompt_suffix="Q1", max_new_tokens=6)
    out_ba = wrapper.generate([c2, c1], prompt_suffix="Q1", max_new_tokens=6)
    assert len(out_ab) == 6 and len(out_ba) == 6
    assert len(bm.free_blocks) == free_after_prefill


def test_stress_2_quantization_and_lse_bridge_selection():
    torch.manual_seed(42)
    x = torch.randn((4, 2, 32, 128), dtype=torch.float32)

    q8, s8 = KVQuantizer.quantize_int8(x, dim=-1)
    x8 = KVQuantizer.dequantize_int8(q8, s8)
    assert F.cosine_similarity(x.flatten(), x8.flatten(), dim=0).item() > 0.999

    q4, s4 = KVQuantizer.quantize_int4(x, dim=-1)
    x4 = KVQuantizer.dequantize_int4(q4, s4)
    assert F.cosine_similarity(x.flatten(), x4.flatten(), dim=0).item() > 0.985

    q_probe = torch.randn((4, 24, 64), dtype=torch.float32)
    k_prior = torch.randn((4, 48, 64), dtype=torch.float32)
    lse_local = torch.randn((4, 24), dtype=torch.float32)
    alpha = compute_lse_perturbation(q_probe, k_prior, lse_local, scale=1.0 / 8.0)

    sel_low = select_bridge_tokens(alpha, top_k_percent=0.10, chunk_global_offset=100)
    sel_part = select_bridge_tokens(alpha, top_k_percent=0.25, chunk_global_offset=100)
    sel_full = select_bridge_tokens(alpha, top_k_percent=1.0, chunk_global_offset=100)
    assert 1 <= sel_low.num_bridge_tokens <= sel_part.num_bridge_tokens < 24
    assert sel_full.num_bridge_tokens == 24


def test_stress_3_kv_pool_exhaustion_and_cow_refcount():
    torch.manual_seed(42)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev.startswith("cuda") else torch.float32

    wrapper = DAGKVModelWrapper(
        model=_make_tiny_qwen(dev, dtype),
        tokenizer=_StressTokenizer(),
        max_num_blocks=48,
        block_size=16,
        quantization="int8",
        bridge_top_k=0.25,
    )
    bm = wrapper.engine.block_manager

    c1 = wrapper.prefill_chunk("A" * 24)
    c2 = wrapper.prefill_chunk("B" * 24)
    c3 = wrapper.prefill_chunk("C" * 24)
    after_canon_free = len(bm.free_blocks)

    perms = [[c1, c2, c3], [c2, c3, c1], [c3, c1, c2], [c1, c3, c2]]
    for i in range(100):
        out = wrapper.generate(perms[i % len(perms)], prompt_suffix="Q" * 10, max_new_tokens=12)
        assert len(out) == 12

    assert len(bm.free_blocks) == after_canon_free
    for b_ids in bm.canonical_registry.values():
        for bid in b_ids:
            assert bm.physical_metadata[bid].ref_count == 1


def test_stress_4_extreme_32k_rope_gqa_and_parity():
    torch.manual_seed(1337)
    block_size, num_blocks, batch_size = 16, 32, 4
    configs = [
        (28, 4, 128, 1000000.0, "none"),
        (28, 4, 128, 1000000.0, "int8"),
        (16, 2, 64, 10000.0, "none"),
        (8, 8, 128, 500000.0, "int8"),
    ]
    extreme_offsets = torch.tensor([
        [-32768, -4096, 0, 4096, 16384, 32768],
        [32768, -16384, 512, -1024, 0, 8192],
        [-8192, 24576, -16, 16, -32768, 32768],
        [0, 32768, -32768, 1234, -5678, 9999],
    ], dtype=torch.int32)
    ragged_valid_lens = torch.tensor([
        [3, 16, 1, 9, 15, 16],
        [16, 7, 16, 2, 11, 4],
        [1, 1, 1, 16, 16, 8],
        [15, 14, 13, 12, 11, 10],
    ], dtype=torch.int32)
    block_tables = torch.arange(24, dtype=torch.int32).reshape(4, 6)
    block_starts = torch.zeros_like(ragged_valid_lens)
    block_starts[:, 1:] = torch.cumsum(ragged_valid_lens[:, :-1], dim=1)
    seq_lens = ragged_valid_lens.sum(dim=1).to(torch.int32)

    for q_heads, kv_heads, head_dim, rope_base, q_mode in configs:
        sm_scale = 1.0 / math.sqrt(head_dim)
        q_cpu = torch.randn((batch_size, q_heads, 1, head_dim), dtype=torch.float16)
        k_fp16 = torch.randn((num_blocks, kv_heads, block_size, head_dim), dtype=torch.float16)
        v_fp16 = torch.randn((num_blocks, kv_heads, block_size, head_dim), dtype=torch.float16)

        if q_mode == "int8":
            k_pool_cpu, k_scales_cpu = KVQuantizer.quantize_int8(k_fp16, dim=-1)
            v_pool_cpu, v_scales_cpu = KVQuantizer.quantize_int8(v_fp16, dim=-1)
        else:
            k_pool_cpu, v_pool_cpu = k_fp16, v_fp16
            k_scales_cpu, v_scales_cpu = None, None

        out_pytorch = adjoint_paged_attention(
            q=q_cpu.float(),
            k_pool=k_pool_cpu if q_mode == "int8" else k_pool_cpu.float(),
            v_pool=v_pool_cpu if q_mode == "int8" else v_pool_cpu.float(),
            block_tables=block_tables,
            offset_tables=extreme_offsets,
            seq_lens=seq_lens,
            sm_scale=sm_scale,
            base=rope_base,
            block_start_tables=block_starts,
            valid_lens_tables=ragged_valid_lens,
            k_scales=k_scales_cpu.float() if k_scales_cpu is not None else None,
            v_scales=v_scales_cpu.float() if v_scales_cpu is not None else None,
        ).float()
        assert not torch.isnan(out_pytorch).any()

        if torch.cuda.is_available():
            out_triton = adjoint_paged_attention(
                q=q_cpu.to("cuda:0"),
                k_pool=k_pool_cpu.to("cuda:0"),
                v_pool=v_pool_cpu.to("cuda:0"),
                block_tables=block_tables.to("cuda:0"),
                offset_tables=extreme_offsets.to("cuda:0"),
                seq_lens=seq_lens.to("cuda:0"),
                sm_scale=sm_scale,
                base=rope_base,
                block_start_tables=block_starts.to("cuda:0"),
                valid_lens_tables=ragged_valid_lens.to("cuda:0"),
                k_scales=k_scales_cpu.to("cuda:0") if k_scales_cpu is not None else None,
                v_scales=v_scales_cpu.to("cuda:0") if v_scales_cpu is not None else None,
            ).cpu().float()
            cos_sim = F.cosine_similarity(out_triton.flatten(), out_pytorch.flatten(), dim=0).item()
            assert cos_sim > 0.999


def test_stress_5_multithreaded_http_server_hammering():
    torch.manual_seed(99)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev.startswith("cuda") else torch.float32

    wrapper = DAGKVModelWrapper(
        model=_make_tiny_qwen(dev, dtype),
        tokenizer=_StressTokenizer(),
        max_num_blocks=128,
        block_size=16,
        quantization="int8",
        bridge_top_k=0.2,
    )
    lock = threading.RLock()
    chunk_cache = {}
    stats = {"hits": 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            full_text = "\n".join(m.get("content", "") for m in payload.get("messages", []))
            parts = re.split(r"(?=### FILE: )", full_text)
            file_segments = [p.strip() + "\n\n" for p in parts if p.startswith("### FILE: ")]
            last_file, q_tail = file_segments[-1].rsplit("\nQuestion:", 1)
            file_segments[-1] = last_file.strip() + "\n\n"
            query_str = "\nQuestion:" + q_tail

            with lock:
                active = []
                for seg in file_segments:
                    h_key = hashlib.sha256(seg.encode("utf-8")).hexdigest()[:16]
                    if h_key in chunk_cache:
                        active.append(chunk_cache[h_key])
                        stats["hits"] += 1
                    else:
                        c = wrapper.prefill_chunk(seg)
                        chunk_cache[h_key] = c
                        active.append(c)
                ans = wrapper.generate(active, prompt_suffix=query_str, max_new_tokens=6).strip()

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"id": uuid.uuid4().hex, "ans": ans}).encode("utf-8"))

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    files = [
        "### FILE: auth.py\ndef verify_token(tok):\n    return tok == 'secret_42'\n",
        "### FILE: db.py\nclass Pool:\n    max_conns = 64\n    timeout_s = 30\n",
        "### FILE: router.py\ndef route(req):\n    return '/v1/chat/completions'\n",
        "### FILE: cache.py\nBLOCK_SIZE = 16\nMAX_BLOCKS = 2048\n",
    ]
    perms = [[0, 1, 2, 3], [3, 2, 1, 0], [1, 3, 0, 2], [2, 0, 3, 1]]

    def send_req(idx):
        perm = perms[idx % len(perms)]
        content = "\n".join(files[i] for i in perm) + f"\nQuestion: Q{idx}?"
        body = json.dumps({"messages": [{"role": "user", "content": content}]}).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            assert resp.status == 200
            return json.loads(resp.read().decode("utf-8"))

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = [f.result() for f in as_completed([pool.submit(send_req, i) for i in range(24)])]

    server.shutdown()

    bm = wrapper.engine.block_manager
    canon_file_blocks = sum(len(v) for v in bm.canonical_registry.values())
    sink_blocks = 1 if wrapper.engine._sink_phys_block is not None else 0
    assert len(results) == 24
    assert stats["hits"] == 92
    assert len(bm.free_blocks) == bm.max_num_blocks - (canon_file_blocks + sink_blocks)


def test_stress_6_dual_gpu_and_exception_rollback():
    torch.manual_seed(77)
    dev0 = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev0.startswith("cuda") else torch.float32

    model_0 = _make_tiny_qwen(dev0, dtype)
    wrap_0 = DAGKVModelWrapper(model=model_0, tokenizer=_StressTokenizer(), max_num_blocks=64, block_size=16, quantization="int8")
    chunk_local = wrap_0.prefill_chunk("Local canonical chunk B " * 3)

    if torch.cuda.device_count() >= 2:
        model_1 = _make_tiny_qwen("cuda:1", torch.float16)
        model_1.load_state_dict(model_0.state_dict())
        wrap_1 = DAGKVModelWrapper(model=model_1, tokenizer=_StressTokenizer(), max_num_blocks=64, block_size=16, quantization="int8")
        chunk_remote = wrap_1.prefill_chunk("Cross-GPU canonical chunk A " * 3)
        out_cross = wrap_0.generate([chunk_remote, chunk_local], prompt_suffix="Query?", max_new_tokens=10)
        assert len(out_cross) == 10

    bm = wrap_0.engine.block_manager
    free_before = len(bm.free_blocks)
    orig_append = wrap_0.engine.append_and_decode
    counter = {"n": 0}

    def crashing_append(table, token_id, current_pos):
        counter["n"] += 1
        if counter["n"] == 3:
            raise RuntimeError("Simulated mid-decode abort")
        return orig_append(table, token_id, current_pos)

    wrap_0.engine.append_and_decode = crashing_append
    try:
        try:
            wrap_0.generate([chunk_local, chunk_local], prompt_suffix="Crash", max_new_tokens=8)
            assert False, "Expected RuntimeError"
        except RuntimeError:
            pass
    finally:
        wrap_0.engine.append_and_decode = orig_append

    assert len(bm.free_blocks) == free_before
    out_rec = wrap_0.generate([chunk_local], prompt_suffix="Recover", max_new_tokens=6)
    assert len(out_rec) == 6
