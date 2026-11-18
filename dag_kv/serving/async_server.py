"""Async OpenAI-Compatible HTTP Serving Endpoint (/v1/chat/completions).

Built on pure standard-library asyncio without heavyweight external dependencies.
Supports concurrent non-blocking streaming (Server-Sent Events) and batched requests.
"""

import asyncio
import json
import time
from typing import Optional
from .scheduler import ContinuousBatchingScheduler


class AsyncServer:
    """Async HTTP server handling continuous batching and SSE streaming."""

    def __init__(
        self,
        scheduler: ContinuousBatchingScheduler,
        host: str = "127.0.0.1",
        port: int = 8000,
    ):
        self.scheduler = scheduler
        self.host = host
        self.port = port
        self.server: Optional[asyncio.Server] = None

    async def start(self):
        """Starts the scheduler loop and binds the TCP socket server."""
        self.scheduler.start()
        self.server = await asyncio.start_server(
            self._handle_client,
            self.host,
            self.port,
        )

    async def stop(self):
        """Stops the TCP server and continuous batching scheduler."""
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
        self.scheduler.stop()

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            line = await reader.readline()
            if not line:
                writer.close()
                await writer.wait_closed()
                return

            req_line = line.decode("utf-8").strip()
            parts = req_line.split()
            if len(parts) < 2:
                writer.close()
                await writer.wait_closed()
                return

            method, path = parts[0], parts[1]
            headers = {}
            while True:
                h_line = await reader.readline()
                if not h_line or h_line == b"\r\n" or h_line == b"\n":
                    break
                decoded_h = h_line.decode("utf-8").strip()
                if ":" in decoded_h:
                    k, v = decoded_h.split(":", 1)
                    headers[k.strip().lower()] = v.strip()

            content_length = int(headers.get("content-length", 0))
            body_bytes = await reader.readexactly(content_length) if content_length > 0 else b""

            if method == "GET" and (path == "/v1/pool/stats" or path == "/stats"):
                bm = self.scheduler.engine.block_manager
                stats = {
                    "total_requests": self.scheduler.total_requests,
                    "total_file_hits": self.scheduler.total_file_hits,
                    "total_file_lookups": self.scheduler.total_file_lookups,
                    "completed_requests": self.scheduler.completed_requests,
                    "running_batch_size": len(self.scheduler.running_batch),
                    "waiting_queue_size": self.scheduler.waiting_queue.qsize(),
                    "free_blocks": len(bm.free_blocks),
                    "evictions": bm.evictions,
                }
                body = json.dumps(stats).encode("utf-8")
                resp = (
                    b"HTTP/1.1 200 OK\r\n"
                    b"Content-Type: application/json\r\n"
                    b"Content-Length: " + str(len(body)).encode("utf-8") + b"\r\n"
                    b"Connection: close\r\n\r\n" + body
                )
                writer.write(resp)
                await writer.drain()

            elif method == "GET" and (path == "/health" or path == "/v1/health"):
                body = b'{"status": "ok"}'
                resp = (
                    b"HTTP/1.1 200 OK\r\n"
                    b"Content-Type: application/json\r\n"
                    b"Content-Length: " + str(len(body)).encode("utf-8") + b"\r\n"
                    b"Connection: close\r\n\r\n" + body
                )
                writer.write(resp)
                await writer.drain()

            elif method == "POST" and path == "/v1/chat/completions":
                payload = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
                messages = payload.get("messages", [])
                max_tokens = int(payload.get("max_tokens", 16))
                temperature = float(payload.get("temperature", 0.0))
                stream = bool(payload.get("stream", False))
                model_name = payload.get("model", "DAG-KV")

                if stream:
                    # Server-Sent Events (SSE) streaming response
                    header = (
                        b"HTTP/1.1 200 OK\r\n"
                        b"Content-Type: text/event-stream\r\n"
                        b"Cache-Control: no-cache\r\n"
                        b"Connection: close\r\n\r\n"
                    )
                    writer.write(header)
                    await writer.drain()

                    req = await self.scheduler.add_request(
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stream=True,
                        model=model_name,
                    )

                    async for token_dict in self.scheduler.stream_tokens(req):
                        if "token" in token_dict:
                            chunk_data = {
                                "id": f"chatcmpl-{req.request_id}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": model_name,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": token_dict["token"]},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            chunk_str = f"data: {json.dumps(chunk_data)}\n\n"
                            writer.write(chunk_str.encode("utf-8"))
                            await writer.drain()

                        if token_dict.get("done"):
                            stop_data = {
                                "id": f"chatcmpl-{req.request_id}",
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": model_name,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {},
                                        "finish_reason": token_dict.get("finish_reason", "stop"),
                                    }
                                ],
                            }
                            writer.write(f"data: {json.dumps(stop_data)}\n\n".encode("utf-8"))
                            writer.write(b"data: [DONE]\n\n")
                            await writer.drain()
                            break
                else:
                    # Non-streaming response
                    resp_dict = await self.scheduler.generate_response(
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        model=model_name,
                    )
                    body = json.dumps(resp_dict).encode("utf-8")
                    resp = (
                        b"HTTP/1.1 200 OK\r\n"
                        b"Content-Type: application/json\r\n"
                        b"Content-Length: " + str(len(body)).encode("utf-8") + b"\r\n"
                        b"Connection: close\r\n\r\n" + body
                    )
                    writer.write(resp)
                    await writer.drain()
            else:
                body = b'{"error": "Not Found"}'
                resp = (
                    b"HTTP/1.1 404 Not Found\r\n"
                    b"Content-Type: application/json\r\n"
                    b"Content-Length: " + str(len(body)).encode("utf-8") + b"\r\n"
                    b"Connection: close\r\n\r\n" + body
                )
                writer.write(resp)
                await writer.drain()

        except Exception as e:
            try:
                err_body = json.dumps({"error": str(e)}).encode("utf-8")
                err_resp = (
                    b"HTTP/1.1 500 Internal Server Error\r\n"
                    b"Content-Type: application/json\r\n"
                    b"Content-Length: " + str(len(err_body)).encode("utf-8") + b"\r\n"
                    b"Connection: close\r\n\r\n" + err_body
                )
                writer.write(err_resp)
                await writer.drain()
            except Exception:
                pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
