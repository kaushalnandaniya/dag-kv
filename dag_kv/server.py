"""DAG-KV OpenAI-Compatible Serving Endpoint (/v1/chat/completions).

Exposes DualGPUSharded7BEngine + LRUCacheManager + Pyramid Bridge Repair over HTTP
so IDE assistants (Cursor, Continue.dev) and multi-agent pipelines can query shared
repository contexts in any file order with 100% canonical KV cache reuse.
"""

import hashlib
import json
import re
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Tuple
import torch




class DAGKVService:
    """Manages canonical file KV blocks, LRU eviction, and DAG stitching for HTTP requests."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", max_num_blocks: int = 768):
        from benchmarks.bench_7b_realworld_rag import DualGPUSharded7BEngine
        from benchmarks.bench_batched_lru_serving import LRUCacheManager
        self.engine = DualGPUSharded7BEngine(model_name=model_name, max_num_blocks=max_num_blocks)
        self.lru = LRUCacheManager(self.engine.bm_0, self.engine.bm_1)
        self.total_requests = 0
        self.total_file_hits = 0
        self.total_file_lookups = 0

    def _parse_files_and_query(self, messages: List[Dict[str, str]]) -> Tuple[List[str], str]:
        full_text = "\n".join(m.get("content", "") for m in messages)
        if "### FILE:" not in full_text:
            return [], full_text

        parts = re.split(r"(?=### FILE: )", full_text)
        file_segments = [p for p in parts if p.startswith("### FILE: ")]
        prefix_non_file = [p for p in parts if not p.startswith("### FILE: ")]

        query_str = "\n".join(prefix_non_file).strip()
        if file_segments and "\nQuestion:" in file_segments[-1]:
            last_file, q_tail = file_segments[-1].rsplit("\nQuestion:", 1)
            file_segments[-1] = last_file
            query_str = "\nQuestion:" + q_tail

        # Normalize whitespace so SHA-256 hash is identical regardless of file position
        file_segments = [seg.strip() + "\n\n" for seg in file_segments]
        return file_segments, query_str

    @torch.inference_mode()
    def chat_completion(self, payload: Dict) -> Dict:
        t_start = time.perf_counter()
        self.total_requests += 1
        messages = payload.get("messages", [])
        max_tokens = int(payload.get("max_tokens", 8))

        file_segments, query_str = self._parse_files_and_query(messages)
        active_chunks = []
        active_keys = []
        hits = 0

        t_prefill_start = time.perf_counter()
        for seg in file_segments:
            h_key = hashlib.sha256(seg.encode("utf-8")).hexdigest()[:16]
            active_keys.append(h_key)
            self.total_file_lookups += 1
            if h_key in self.lru.entries and not self.lru.entries[h_key].get("evicted", False):
                self.lru.touch(h_key)
                active_chunks.append(self.lru.entries[h_key])
                hits += 1
                self.total_file_hits += 1
            else:
                est_blocks = max(32, len(self.engine.tokenizer.encode(seg, add_special_tokens=False)) // 16 + 4)
                self.lru.ensure_capacity(blocks_needed=est_blocks, protected_keys=active_keys)
                chunk = self.engine.prefill_canonical_chunk(seg)
                self.lru.register(h_key, chunk)
                active_chunks.append(chunk)

        torch.cuda.synchronize("cuda:0")
        torch.cuda.synchronize("cuda:1")
        prefill_ms = (time.perf_counter() - t_prefill_start) * 1000.0

        t_stitch_start = time.perf_counter()
        req_id = f"req_{uuid.uuid4().hex[:8]}"
        table, doc_tokens = self.engine.assemble_prompt_dag(active_chunks, seq_id=req_id)
        self.engine.repair_bridge_tokens(table, active_chunks, top_k_percent=0.08)

        # Decode user query tokens directly against the stitched multi-file KV table
        q_ids = self.engine.tokenizer.encode(query_str, add_special_tokens=False)
        cur_pos = len(doc_tokens)
        for q_tok in q_ids:
            logits = self.engine.append_and_decode(table, token_id=q_tok, current_pos=cur_pos)
            cur_pos += 1

        gen_ids = []
        for step in range(max_tokens):
            next_tok = int(torch.argmax(logits, dim=-1).item())
            gen_ids.append(next_tok)
            tok_str = self.engine.tokenizer.decode([next_tok])
            if next_tok == self.engine.tokenizer.eos_token_id or "\n" in tok_str or step == max_tokens - 1:
                break
            logits = self.engine.append_and_decode(table, token_id=next_tok, current_pos=cur_pos)
            cur_pos += 1

        torch.cuda.synchronize("cuda:0")
        torch.cuda.synchronize("cuda:1")
        stitch_decode_ms = (time.perf_counter() - t_stitch_start) * 1000.0
        total_ms = (time.perf_counter() - t_start) * 1000.0

        from benchmarks.bench_multiuser_codebase import release_request_blocks
        release_request_blocks(self.engine, table)

        raw_ans = self.engine.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        answer_text = raw_ans.split()[0].rstrip(".,;:") if raw_ans else ""
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:10]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": payload.get("model", "Qwen/Qwen2.5-7B-Instruct-DAG-KV"),
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": len(doc_tokens) + len(q_ids),
                "completion_tokens": len(gen_ids),
                "total_tokens": len(doc_tokens) + len(q_ids) + len(gen_ids),
            },
            "dag_kv_telemetry": {
                "files_attached": len(file_segments),
                "canonical_file_hits": hits,
                "canonical_hit_rate": round(hits / max(1, len(file_segments)) * 100.0, 1),
                "prefill_ms": round(prefill_ms, 1),
                "stitch_and_decode_ms": round(stitch_decode_ms, 1),
                "total_latency_ms": round(total_ms, 1),
                "lru_evictions_total": self.lru.evictions,
            },
        }


def create_http_server(service: DAGKVService, host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # Keep benchmark console clean

        def do_GET(self):
            if self.path == "/v1/pool/stats":
                body = json.dumps({
                    "total_requests": service.total_requests,
                    "total_file_hits": service.total_file_hits,
                    "total_file_lookups": service.total_file_lookups,
                    "free_blocks": len(service.engine.bm_0.free_blocks),
                    "lru_evictions": service.lru.evictions,
                }).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

        def do_POST(self):
            if self.path == "/v1/chat/completions":
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                resp = service.chat_completion(payload)
                body = json.dumps(resp).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

    return ThreadingHTTPServer((host, port), Handler)
