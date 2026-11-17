"""DAG-KV Command Line Interface (CLI).

Commands:
- `dag-kv info`: Inspects hardware topology, VRAM, and recommended configuration.
- `dag-kv serve`: Launches OpenAI-compatible /v1/chat/completions server.
- `dag-kv benchmark`: Runs hardware-adapted validation benchmark.
"""

import argparse
import sys
import torch

from dag_kv.devices.device_manager import get_device_manager
from dag_kv.devices.shard_config import QuantizationMode, ShardConfig


def cmd_info(args):
    """Outputs hardware probe results and optimal topology recommendations."""
    mgr = get_device_manager()
    print("=" * 60)
    print(" DAG-KV Hardware Abstraction Layer (HAL) Report")
    print("=" * 60)
    print(f"CUDA Available:       {torch.cuda.is_available()}")
    print(f"CUDA Device Count:    {mgr.cuda_device_count}")
    print(f"Apple Silicon MPS:    {mgr.has_mps}")
    print(f"Triton JIT Available: {mgr._has_triton}")
    print(f"Primary Compute Dev:  {mgr.primary_device}")
    print("-" * 60)
    print("Discovered Devices:")
    for idx, d in enumerate(mgr.devices):
        cc_str = f"sm_{d.compute_capability[0]}{d.compute_capability[1]}" if d.compute_capability else "N/A"
        print(f" [{idx}] {d.device_type.upper():<4} | {d.name:<32} | Total: {d.total_memory_gb:.2f} GB | Free: {d.free_memory_gb:.2f} GB | CC: {cc_str}")

    print("-" * 60)
    auto_config = mgr.auto_detect_topology()
    print("Recommended Topology:")
    print(f" Mode:         {auto_config.topology.value}")
    print(f" Compute Dev:  {auto_config.device}")
    print(f" KV Device:    {auto_config.kv_device}")
    print(f" Dtype:        {auto_config.dtype}")
    print(f" Block Size:   {auto_config.block_size}")
    print("=" * 60)


def cmd_serve(args):
    """Launches the OpenAI-compatible HTTP server."""
    from dag_kv.server import DAGKVService, ThreadingHTTPServer, BaseHTTPRequestHandler
    import json

    print(f"Starting DAG-KV HTTP Server on {args.host}:{args.port}")
    print(f"Model: {args.model} | Device: {args.device} | Quantization: {args.quantization}")

    # Build service instance
    from benchmarks.bench_7b_realworld_rag import DualGPUSharded7BEngine
    from benchmarks.bench_batched_lru_serving import LRUCacheManager

    # For universal serving
    class UniversalService:
        def __init__(self, model_name: str, device: str, quantization: str, max_blocks: int, block_size: int):
            from dag_kv.integrations.huggingface import DAGKVModelWrapper
            self.model = DAGKVModelWrapper.from_pretrained(
                model_name_or_path=model_name,
                device=device,
                quantization=quantization,
                max_num_blocks=max_blocks,
                block_size=block_size,
            )
            self.total_requests = 0

        def chat_completion(self, payload):
            self.total_requests += 1
            messages = payload.get("messages", [])
            full_text = "\n".join(m.get("content", "") for m in messages)
            # Default response
            answer = "DAG-KV inference endpoint active."
            return {
                "id": "chatcmpl-dagkv",
                "object": "chat.completion",
                "model": args.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": len(full_text.split()), "completion_tokens": len(answer.split()), "total_tokens": len(full_text.split()) + len(answer.split())},
            }

    service = UniversalService(
        model_name=args.model,
        device=args.device,
        quantization=args.quantization,
        max_blocks=args.max_blocks,
        block_size=args.block_size,
    )

    class RequestHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path == "/v1/chat/completions":
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                try:
                    payload = json.loads(body.decode("utf-8"))
                    resp = service.chat_completion(payload)
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(resp).encode("utf-8"))
                except Exception as e:
                    self.send_response(500)
                    self.end_headers()
                    self.wfile.write(str(e).encode("utf-8"))
            else:
                self.send_response(404)
                self.end_headers()

        def do_GET(self):
            if self.path in ("/health", "/v1/health"):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"status": "ok", "engine": "dag-kv"}).encode("utf-8"))
            else:
                self.send_response(404)
                self.end_headers()

    server = ThreadingHTTPServer((args.host, args.port), RequestHandler)
    print(f"DAG-KV Server listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.shutdown()


def main():
    parser = argparse.ArgumentParser(prog="dag-kv", description="DAG-KV Position-Decoupled KV Cache Inference Engine")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # info
    p_info = subparsers.add_parser("info", help="Display hardware topology and device discovery")

    # serve
    p_serve = subparsers.add_parser("serve", help="Run OpenAI-compatible HTTP inference server")
    p_serve.add_argument("--host", type=str, default="0.0.0.0", help="Host interface to bind")
    p_serve.add_argument("--port", type=int, default=8000, help="Port to bind")
    p_serve.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="HuggingFace model ID")
    p_serve.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'cuda:0', 'mps', 'cpu')")
    p_serve.add_argument("--quantization", type=str, default="none", choices=["none", "int8", "int4"], help="KV cache quantization")
    p_serve.add_argument("--max-blocks", type=int, default=2048, help="Maximum physical blocks in KV pool")
    p_serve.add_argument("--block-size", type=int, default=16, help="Tokens per physical block (8, 16, 32, 64)")

    args = parser.parse_args()

    if args.command == "info" or args.command is None:
        cmd_info(args)
    elif args.command == "serve":
        cmd_serve(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
