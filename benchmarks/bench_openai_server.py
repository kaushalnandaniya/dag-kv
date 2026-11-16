"""Upgrade 4 Verification: End-to-End OpenAI `/v1/chat/completions` Live Server Test."""

import json
import threading
import urllib.request

from benchmarks.bench_multiuser_codebase import load_real_codebase_chunk
from dag_kv.server import DAGKVService, create_http_server


def post_chat_completion(url: str, prompt: str, max_tokens: int = 4) -> dict:
    req_data = json.dumps({
        "model": "Qwen/Qwen2.5-7B-Instruct-DAG-KV",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
    }).encode("utf-8")
    req = urllib.request.Request(url, data=req_data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def run_server_benchmark():
    print("\n==========================================================================================")
    print("  UPGRADE 4: LIVE OPENAI-COMPATIBLE SERVER (/v1/chat/completions on 2x RTX 2080 Ti)")
    print("==========================================================================================\n")

    service = DAGKVService(model_name="Qwen/Qwen2.5-7B-Instruct", max_num_blocks=768)
    httpd = create_http_server(service, host="127.0.0.1", port=18080)
    server_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    server_thread.start()
    endpoint = "http://127.0.0.1:18080/v1/chat/completions"

    f_chunks = {
        "rope": load_real_codebase_chunk("dag_kv/core/rope_adjoint.py"),
        "sink": load_real_codebase_chunk("dag_kv/core/sink_canonicalizer.py"),
        "lse": load_real_codebase_chunk("dag_kv/core/lse_probe.py"),
        "block": load_real_codebase_chunk("dag_kv/core/block_manager.py"),
    }

    requests_suite = [
        (
            "Req 1 (Cold Index)",
            ["rope", "sink", "lse", "block"],
            "\nQuestion: In dag_kv/core/block_manager.py, what is the default integer value of max_num_blocks in BlockManager.__init__? Answer with ONLY the number:\nAnswer:",
            "4096",
        ),
        (
            "Req 2 (Permuted #1)",
            ["block", "lse", "rope", "sink"],
            "\nQuestion: In dag_kv/core/sink_canonicalizer.py, what is the default integer value of num_sink_tokens in SinkCanonicalizer.__init__? Answer with ONLY the number:\nAnswer:",
            "4",
        ),
        (
            "Req 3 (Permuted #2)",
            ["sink", "block", "rope", "lse"],
            "\nQuestion: In dag_kv/core/lse_probe.py, what mathematical activation function is applied to raw_diff in compute_lse_perturbation? Answer with ONLY the lowercase function name:\nAnswer:",
            "sigmoid",
        ),
        (
            "Req 4 (Permuted #3)",
            ["lse", "sink", "block", "rope"],
            "\nQuestion: In dag_kv/core/rope_adjoint.py, what is the default float value of base in compute_inv_freq? Answer with ONLY the number:\nAnswer:",
            "10000.0",
        ),
    ]

    print(f"{'HTTP Client Request':<20} | {'File Order':<24} | {'Hit%':<6} | {'Expect':<8} | {'Pred':<8} | {'Total ms':<9} | {'Speedup'}")
    print("-" * 94)

    cold_ms = None
    for label, order, q_str, exp in requests_suite:
        prompt = "".join(f_chunks[k] for k in order) + q_str
        resp = post_chat_completion(endpoint, prompt, max_tokens=10)
        pred = resp["choices"][0]["message"]["content"]
        tel = resp["dag_kv_telemetry"]
        tot_ms = tel["total_latency_ms"]
        if cold_ms is None:
            cold_ms = tot_ms
        sp = cold_ms / tot_ms
        print(f"{label:<20} | {str(order):<24} | {tel['canonical_hit_rate']:>5.1f}% | {exp:<8} | {pred:<8} | {tot_ms:>7.1f}ms | {sp:>5.2f}x")

    httpd.shutdown()
    print("-" * 94 + "\n")


if __name__ == "__main__":
    run_server_benchmark()
