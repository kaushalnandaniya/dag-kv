"""Continuous Batching Async Scheduler for GPU DAG-KV Serving.

Replaces synchronous request-level locking with true ragged multi-tenant
continuous batching: concurrent decode steps across heterogeneous sequences
are scheduled and evaluated in a single forward pass per iteration on the GPU.
"""

import asyncio
from dataclasses import dataclass, field
import hashlib
import json
import re
import time
from typing import AsyncIterator, Dict, List, Optional, Tuple
import torch

from dag_kv.core.block_manager import OffsetBlockTable
from dag_kv.core.sink_canonicalizer import CanonicalChunk
from dag_kv.models.base_engine import DAGKVEngine


@dataclass
class RequestState:
    """State tracking for a single in-flight continuous batching request."""
    request_id: str
    file_segments: List[str]
    query_str: str
    max_new_tokens: int = 16
    temperature: float = 0.0
    stream: bool = False
    model_name: str = "default"

    # Runtime state
    active_chunks: List[CanonicalChunk] = field(default_factory=list)
    table: Optional[OffsetBlockTable] = None
    current_pos: int = 0
    last_token: int = 0
    prompt_tokens_len: int = 0
    generated_tokens: List[int] = field(default_factory=list)
    output_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    status: str = "PENDING"  # PENDING, RUNNING, FINISHED, ERROR
    error: Optional[Exception] = None
    finish_reason: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    prefill_ms: float = 0.0
    decode_ms: float = 0.0


def parse_files_and_query(messages: List[Dict[str, str]]) -> Tuple[List[str], str]:
    """Extracts ### FILE: segments and user prompt query from OpenAI chat messages."""
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

    file_segments = [seg.strip() + "\n\n" for seg in file_segments]
    return file_segments, query_str


class ContinuousBatchingScheduler:
    """Async scheduler executing continuous batching on GPU across concurrent requests."""

    def __init__(
        self,
        engine: DAGKVEngine,
        max_batch_size: int = 16,
        step_interval_s: float = 0.002,
    ):
        self.engine = engine
        self.max_batch_size = max_batch_size
        self.step_interval_s = step_interval_s

        self.waiting_queue: asyncio.Queue[RequestState] = asyncio.Queue()
        self.running_batch: List[RequestState] = []
        self._loop_task: Optional[asyncio.Task] = None
        self._running: bool = False

        # Telemetry
        self.total_requests = 0
        self.total_file_lookups = 0
        self.total_file_hits = 0
        self.completed_requests = 0
        self._assistant_suffix: Optional[str] = None
        self._eos_token_ids = self._resolve_eos_token_ids()

    def start(self):
        """Starts the continuous batching worker loop."""
        if not self._running:
            self._running = True
            self._loop_task = asyncio.create_task(self._scheduler_loop())

    def stop(self):
        """Stops the scheduler loop gracefully."""
        self._running = False
        if self._loop_task is not None:
            self._loop_task.cancel()

    async def add_request(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 16,
        temperature: float = 0.0,
        stream: bool = False,
        model: str = "default",
    ) -> RequestState:
        """Enqueues a new client request into the continuous batching scheduler."""
        import uuid
        req_id = f"req_{uuid.uuid4().hex[:10]}"
        files, query = parse_files_and_query(messages)

        req = RequestState(
            request_id=req_id,
            file_segments=files,
            query_str=query,
            max_new_tokens=max_tokens,
            temperature=temperature,
            stream=stream,
            model_name=model,
        )
        self.total_requests += 1
        await self.waiting_queue.put(req)
        return req

    async def stream_tokens(self, req: RequestState) -> AsyncIterator[Dict]:
        """Async generator yielding token deltas as they are computed on the GPU."""
        while True:
            item = await req.output_queue.get()
            if item.get("error"):
                raise item["error"]
            yield item
            if item.get("done", False):
                break

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 16,
        temperature: float = 0.0,
        model: str = "default",
    ) -> Dict:
        """High-level non-streaming OpenAI-compatible completion."""
        req = await self.add_request(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
            model=model,
        )
        full_text = []
        finish_reason = "stop"
        async for token_dict in self.stream_tokens(req):
            if "token" in token_dict:
                full_text.append(token_dict["token"])
            if token_dict.get("done"):
                finish_reason = token_dict.get("finish_reason", "stop")

        ans = "".join(full_text).strip()
        return {
            "id": f"chatcmpl-{req.request_id}",
            "object": "chat.completion",
            "created": int(req.created_at),
            "model": req.model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": ans},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": req.prompt_tokens_len,
                "completion_tokens": len(req.generated_tokens),
                "total_tokens": req.prompt_tokens_len + len(req.generated_tokens),
            },
            "dag_kv_telemetry": {
                "files_attached": len(req.file_segments),
                "prefill_ms": round(req.prefill_ms, 2),
                "decode_ms": round(req.decode_ms, 2),
                "total_ms": round((time.time() - req.created_at) * 1000.0, 2),
            },
        }

    def _resolve_eos_token_ids(self) -> set:
        """Collects all valid EOS/stop token IDs from tokenizer and generation_config."""
        eos_ids = set()
        candidates = [getattr(self.engine.tokenizer, "eos_token_id", None)]
        gen_cfg = getattr(getattr(self.engine, "model", None), "generation_config", None)
        if gen_cfg is not None:
            candidates.append(getattr(gen_cfg, "eos_token_id", None))
        for item in candidates:
            if isinstance(item, int):
                eos_ids.add(item)
            elif isinstance(item, (list, tuple, set)):
                eos_ids.update(x for x in item if isinstance(x, int))
        return eos_ids

    def _get_assistant_turn_suffix(self) -> str:
        """Derives the user-close + assistant-open turn marker from the tokenizer's chat template."""
        if self._assistant_suffix is not None:
            return self._assistant_suffix
        tok = self.engine.tokenizer
        suffix = ""
        if getattr(tok, "chat_template", None) and hasattr(tok, "apply_chat_template"):
            try:
                sentinel = "__DAGKV_TURN_SPLIT__"
                rendered = tok.apply_chat_template(
                    [{"role": "user", "content": sentinel}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if sentinel in rendered:
                    suffix = rendered.split(sentinel, 1)[1]
            except Exception:
                suffix = ""
        self._assistant_suffix = suffix
        return suffix

    async def _emit_token_and_check_done(self, req: RequestState) -> bool:
        """Pushes req.last_token to output_queue and returns True if request finished."""
        next_tok = req.last_token
        tok_str = self.engine.tokenizer.decode([next_tok])
        is_eos = next_tok in self._eos_token_ids
        reached_limit = len(req.generated_tokens) >= req.max_new_tokens

        if is_eos or reached_limit:
            req.status = "FINISHED"
            req.finish_reason = "stop" if is_eos else "length"
            if not is_eos:
                await req.output_queue.put({"token": tok_str, "token_id": next_tok, "done": False})
            await req.output_queue.put({"done": True, "finish_reason": req.finish_reason})
            if req.table is not None:
                self.engine.block_manager.free_sequence(req.table)
                req.table = None
            self.completed_requests += 1
            return True

        await req.output_queue.put({"token": tok_str, "token_id": next_tok, "done": False})
        return False

    async def _scheduler_loop(self):
        """Infinite worker loop advancing admitted requests on each step."""
        while self._running:
            try:
                # 1. Admit new requests up to max_batch_size without starving the event loop
                while len(self.running_batch) < self.max_batch_size and not self.waiting_queue.empty():
                    req = self.waiting_queue.get_nowait()
                    try:
                        await asyncio.to_thread(self._prefill_request, req)
                        finished = await self._emit_token_and_check_done(req)
                        if not finished:
                            self.running_batch.append(req)
                        await asyncio.sleep(0)
                    except Exception as e:
                        req.status = "ERROR"
                        req.error = e
                        if req.table is not None:
                            try:
                                self.engine.block_manager.free_sequence(req.table)
                                req.table = None
                            except Exception:
                                pass
                        await req.output_queue.put({"error": e, "done": True})

                if not self.running_batch:
                    await asyncio.sleep(self.step_interval_s)
                    continue

                # 2. Execute one batched decode step off the event-loop thread
                t0 = time.perf_counter()
                await asyncio.to_thread(self._batched_decode_step)
                step_ms = (time.perf_counter() - t0) * 1000.0

                # 3. Emit tokens and flush SSE writers immediately
                still_running = []
                for req in self.running_batch:
                    req.decode_ms += step_ms
                    finished = await self._emit_token_and_check_done(req)
                    if not finished:
                        still_running.append(req)

                self.running_batch = still_running
                await asyncio.sleep(0)

            except asyncio.CancelledError:
                break
            except Exception as e:
                for req in self.running_batch:
                    if req.table is not None:
                        try:
                            self.engine.block_manager.free_sequence(req.table)
                            req.table = None
                        except Exception:
                            pass
                    await req.output_queue.put({"error": e, "done": True})
                self.running_batch.clear()
                await asyncio.sleep(self.step_interval_s)

    def _sample_next_token(self, logits: torch.Tensor, temperature: float) -> int:
        if temperature <= 0.0:
            return int(torch.argmax(logits, dim=-1).item())
        probs = torch.softmax(logits / temperature, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())

    def _prefill_request(self, req: RequestState):
        """Assembles prompt DAG and runs micro-prefill repair for newly admitted request."""
        t_start = time.perf_counter()
        active_chunks = []
        disk_store = getattr(self.engine.block_manager, "disk_store", None)

        for seg in req.file_segments:
            self.total_file_lookups += 1
            tok_ids = torch.tensor(
                self.engine.tokenizer.encode(seg, add_special_tokens=False),
                dtype=torch.int64,
                device=self.engine.device,
            )
            h_key = self.engine.canonicalizer.compute_chunk_hash(tok_ids)
            in_ram = h_key in self.engine._chunk_cache
            in_disk = (
                not in_ram
                and disk_store is not None
                and hasattr(disk_store, "_get_chunk_path")
                and disk_store._get_chunk_path(h_key).exists()
            )
            if in_ram or in_disk:
                self.total_file_hits += 1
            chunk = self.engine.prefill_canonical_chunk(tok_ids)
            active_chunks.append(chunk)

        table, doc_tokens = self.engine.assemble_prompt_dag(active_chunks, seq_id=req.request_id)
        if len(active_chunks) > 1:
            self.engine.repair_bridge_tokens(table, active_chunks)

        # Decode user query prompt suffix tokens sequentially to prepare decode head
        turn_suffix = self._get_assistant_turn_suffix()
        query_text = req.query_str or ""
        if turn_suffix and not query_text.endswith(turn_suffix):
            query_text = query_text + turn_suffix
        q_ids = self.engine.tokenizer.encode(query_text, add_special_tokens=False) if query_text else []
        cur_pos = len(doc_tokens)
        last_logits = None

        for q_tok in q_ids:
            last_logits = self.engine.append_and_decode(table, token_id=q_tok, current_pos=cur_pos)
            cur_pos += 1

        if last_logits is None:
            init_tok = doc_tokens[-1] if doc_tokens else 0
            last_logits = self.engine.append_and_decode(table, token_id=init_tok, current_pos=cur_pos)
            cur_pos += 1

        next_tok = self._sample_next_token(last_logits, req.temperature)

        req.table = table
        req.active_chunks = active_chunks
        req.current_pos = cur_pos
        req.last_token = next_tok
        req.prompt_tokens_len = len(doc_tokens) + len(q_ids)
        req.generated_tokens.append(next_tok)
        req.status = "RUNNING"
        req.prefill_ms = (time.perf_counter() - t_start) * 1000.0

    def _batched_decode_step(self):
        """Executes a single GPU forward pass batched across all active running sequences."""
        B = len(self.running_batch)
        if B == 0:
            return

        tables = [r.table for r in self.running_batch]
        tokens = [r.last_token for r in self.running_batch]
        positions = [r.current_pos for r in self.running_batch]

        logits_b = self.engine.batched_append_and_decode(tables, tokens, positions)

        for i, req in enumerate(self.running_batch):
            next_tok = self._sample_next_token(logits_b[i], req.temperature)
            req.last_token = next_tok
            req.current_pos += 1
            req.generated_tokens.append(next_tok)
