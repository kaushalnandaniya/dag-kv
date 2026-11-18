"""DAG-KV Command Line Interface (CLI).

Commands:
- `dag-kv info`: Inspects hardware topology, VRAM, and recommended configuration.
- `dag-kv serve`: Launches OpenAI-compatible /v1/chat/completions server.
"""

import argparse
import hashlib
import json
import re
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import torch

from dag_kv.devices.device_manager import get_device_manager


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
    """Launches the universal OpenAI-compatible HTTP server."""
    if getattr(args, "continuous_batching", True):
        import asyncio
        from dag_kv.serving import ContinuousBatchingScheduler, AsyncServer
        from dag_kv.models import AutoDAGKVEngine
        from dag_kv.persistence import DiskKVCache

        disk_store = DiskKVCache(args.cache_dir) if (getattr(args, "persist", False) or getattr(args, "cache_dir", None)) else None
        print(f"Starting DAG-KV Continuous Batching Server on {args.host}:{args.port}")
        print(f"Model: {args.model} | Device: {args.device} | Quantization: {args.quantization} | Max Batch: {args.batch_size}")

        target_dev = None if args.device == "auto" else args.device
        engine = AutoDAGKVEngine.from_pretrained(
            args.model,
            device=target_dev,
            quantization=args.quantization,
            max_num_blocks=args.max_blocks,
            block_size=args.block_size,
            disk_store=disk_store,
        )
        scheduler = ContinuousBatchingScheduler(engine=engine, max_batch_size=args.batch_size)
        server = AsyncServer(scheduler=scheduler, host=args.host, port=args.port)

        async def _run():
            await server.start()
            print(f"DAG-KV Server listening on http://{args.host}:{args.port}")
            try:
                while True:
                    await asyncio.sleep(3600)
            except (asyncio.CancelledError, KeyboardInterrupt):
                pass
            finally:
                await server.stop()

        try:
            asyncio.run(_run())
        except KeyboardInterrupt:
            print("\nShutting down server...")
        return

    from dag_kv.integrations.huggingface import DAGKVModelWrapper

    print(f"Starting DAG-KV HTTP Server on {args.host}:{args.port}")
    print(f"Model: {args.model} | Device: {args.device} | Quantization: {args.quantization}")

    class UniversalService:
        def __init__(self, model_name: str, device: str, quantization: str, max_blocks: int, block_size: int):
            self.model = DAGKVModelWrapper.from_pretrained(
                model_name_or_path=model_name,
                device=device,
                quantization=quantization,
                max_num_blocks=max_blocks,
                block_size=block_size,
            )
            self.chunk_cache = {}
            self.lock = threading.RLock()
            self.total_requests = 0
            self.total_file_hits = 0

        def _parse_files_and_query(self, messages):
            full_text = "\n".join(m.get("content", "") for m in messages)
            if "### FILE:" not in full_text:
                return [], full_text
            parts = re.split(r"(?=### FILE: )", full_text)
            file_segments = [p.strip() + "\n\n" for p in parts if p.startswith("### FILE: ")]
            non_file = [p for p in parts if not p.startswith("### FILE: ")]
            query_str = "\n".join(non_file).strip()
            if file_segments and "\nQuestion:" in file_segments[-1]:
                last_file, q_tail = file_segments[-1].rsplit("\nQuestion:", 1)
                file_segments[-1] = last_file.strip() + "\n\n"
                query_str = "\nQuestion:" + q_tail
            return file_segments, query_str

        def chat_completion(self, payload):
            t0 = time.perf_counter()
            self.total_requests += 1
            messages = payload.get("messages", [])
            max_tokens = int(payload.get("max_tokens", 32))
            file_segments, query_str = self._parse_files_and_query(messages)

            active_chunks = []
            hits = 0
            with self.lock:
                for seg in file_segments:
                    h_key = hashlib.sha256(seg.encode("utf-8")).hexdigest()[:16]
                    if h_key in self.chunk_cache:
                        active_chunks.append(self.chunk_cache[h_key])
                        hits += 1
                        self.total_file_hits += 1
                    else:
                        chunk = self.model.prefill_chunk(seg)
                        self.chunk_cache[h_key] = chunk
                        active_chunks.append(chunk)

                if not active_chunks:
                    chunk = self.model.prefill_chunk(query_str or "Hello")
                    active_chunks = [chunk]
                    query_str = ""

                answer = self.model.generate(
                    chunks=active_chunks,
                    prompt_suffix=query_str,
                    max_new_tokens=max_tokens,
                ).strip()
            total_ms = round((time.perf_counter() - t0) * 1000.0, 1)

            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:10]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": args.model,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": len(query_str.split()), "completion_tokens": len(answer.split()), "total_tokens": len(query_str.split()) + len(answer.split())},
                "dag_kv_telemetry": {
                    "files_attached": len(file_segments),
                    "canonical_file_hits": hits,
                    "total_latency_ms": total_ms,
                },
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

    subparsers.add_parser("info", help="Display hardware topology and device discovery")

    p_serve = subparsers.add_parser("serve", help="Run OpenAI-compatible HTTP inference server")
    p_serve.add_argument("--host", type=str, default="0.0.0.0", help="Host interface to bind")
    p_serve.add_argument("--port", type=int, default=8000, help="Port to bind")
    p_serve.add_argument("--model", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="HuggingFace model ID")
    p_serve.add_argument("--device", type=str, default="auto", help="Compute device ('auto', 'cuda:0', 'mps', 'cpu')")
    p_serve.add_argument("--quantization", type=str, default="none", choices=["none", "int8", "int4"], help="KV cache quantization")
    p_serve.add_argument("--max-blocks", type=int, default=2048, help="Maximum physical blocks in KV pool")
    p_serve.add_argument("--block-size", type=int, default=16, help="Tokens per physical block (8, 16, 32, 64)")
    p_serve.add_argument("--continuous-batching", action="store_true", default=True, help="Enable async continuous batching scheduler")
    p_serve.add_argument("--batch-size", type=int, default=16, help="Max batch size for continuous batching")
    p_serve.add_argument("--cache-dir", type=str, default=None, help="NVMe persistent cache directory")
    p_serve.add_argument("--persist", action="store_true", default=False, help="Enable disk KV persistence")

    args = parser.parse_args()

    if args.command == "info" or args.command is None:
        cmd_info(args)
    elif args.command == "serve":
        cmd_serve(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
