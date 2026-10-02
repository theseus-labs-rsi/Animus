"""Real localhost HTTP fault injection. No external provider or paid requests."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import os
import socket
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
os.environ.setdefault("OPENAI_API_KEY", "offline-only")
os.environ.setdefault("MODEL", "offline-only")
import config
from llm_trace import trace_scope
from llm_transport import original_run_transport
from pipeline.run import Tracer
from types import SimpleNamespace


class LocalServer(ThreadingHTTPServer):
    daemon_threads = True


class HttpLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.requests, self.disconnected = [], threading.Event()
        self.mode = "success"
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "trace.jsonl"
        test = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def handle(self):
                try:
                    super().handle()
                except (ConnectionResetError, ConnectionAbortedError):
                    test.disconnected.set()

            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                test.requests.append(request)
                try:
                    if test.mode in {"slow_headers", "slow_body"}:
                        if test.mode == "slow_body":
                            self.send_response(200)
                            self.send_header("Content-Type", "application/json")
                            self.send_header("Content-Length", "10000")
                            self.end_headers(); self.wfile.write(b"{"); self.wfile.flush()
                        self.connection.settimeout(5)
                        if self.connection.recv(1) == b"":
                            test.disconnected.set()
                        self.close_connection = True
                        return
                    body = json.dumps({"id": "localhost-response", "model": request["model"],
                        "object": "chat.completion", "created": 1,
                        "choices": [{"index": 0, "finish_reason": "stop", "message": {
                            "role": "assistant", "content": '{"ok":true}'}}],
                        "usage": {"prompt_tokens": 17, "completion_tokens": 19, "total_tokens": 36,
                                  "completion_tokens_details": {"reasoning_tokens": 8}}}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    if test.mode == "trickle":
                        for value in body:
                            self.wfile.write(bytes([value])); self.wfile.flush(); time.sleep(0.025)
                    else:
                        self.wfile.write(body); self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    test.disconnected.set()
                except socket.timeout:
                    pass

        self.server = LocalServer(("127.0.0.1", 0), Handler)
        self.worker = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.worker.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        self.patches = patch.multiple(config, API_KEY="offline-only", BASE_URL=f"http://127.0.0.1:{self.server.server_port}/v1",
                                      _LLM_SEM=threading.BoundedSemaphore(1))
        self.patches.start(); self.addCleanup(self.patches.stop)
        original_connect = socket.socket.connect
        def local_only(sock, address):
            if address[0] not in {"127.0.0.1", "::1"}:
                raise AssertionError("External network forbidden")
            return original_connect(sock, address)
        guard = patch.object(socket.socket, "connect", local_only)
        guard.start(); self.addCleanup(guard.stop)

    def records(self):
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines()]

    def invoke(self, deadline=5, read_timeout=5):
        transport = original_run_transport("glm-5.3-flash", "low", read_timeout)
        transport["profiles"]["low"]["deadline_seconds"] = deadline
        with trace_scope(self.path, "world.structure"):
            return config.chat_json([{"role": "user", "content": "Local fixture"}], model="glm-5.3-flash",
                transport=transport, max_tokens=32768, max_output_tokens=65536,
                retries=5, strict_json=True, retry_delay_base=0)

    def test_actual_http_wire_headers_bytes_complete_usage_and_client_close(self):
        self.assertEqual(self.invoke(), {"ok": True})
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.requests[0]["reasoning_effort"], "low")
        self.assertEqual(self.requests[0]["max_tokens"], 32768)
        self.assertNotIn("stream", self.requests[0])
        rows = self.records(); names = [row["event"] for row in rows]
        self.assertLess(names.index("http_response_headers"), names.index("http_body_first_bytes"))
        self.assertLess(names.index("request_client_closed"), names.index("response"))
        response = next(row["response"] for row in rows if row["event"] == "response")
        self.assertEqual(response["usage"]["completion_tokens_details"]["reasoning_tokens"], 8)
        self.assertTrue(next(row for row in rows if row["event"] == "request_client_closed")["local_cleanup_confirmed"])

    def test_slow_headers_total_deadline_closes_socket_before_failure_and_never_retries(self):
        self.mode = "slow_headers"
        started = time.monotonic()
        with self.assertRaises(config.ChatJSONError) as raised:
            self.invoke(deadline=1.5)
        self.assertLess(time.monotonic() - started, 3)
        self.assertEqual(raised.exception.kind, "total_deadline")
        self.assertEqual(raised.exception.phase, "awaiting_http_headers")
        self.assertTrue(raised.exception.local_cleanup_confirmed)
        self.assertEqual(raised.exception.remote_cancellation, "unknown")
        self.assertTrue(self.disconnected.wait(1), "Server did not observe peer socket closing")
        self.assertEqual(len(self.requests), 1)
        names = [row["event"] for row in self.records()]
        self.assertLess(names.index("request_client_closed"), names.index("call_error"))
        self.assertNotIn("response", names)
        self.mode = "success"
        self.assertEqual(self.invoke(), {"ok": True}, "Permit must be reusable after cleanup")

    def test_partial_response_timeout_does_not_become_parse_success(self):
        self.mode = "slow_body"
        with self.assertRaises(config.ChatJSONError) as raised:
            self.invoke(deadline=1.5)
        self.assertEqual(raised.exception.phase, "reading_http_body")
        self.assertTrue(self.disconnected.wait(1))
        self.assertEqual(len(self.requests), 1)
        names = [row["event"] for row in self.records()]
        self.assertIn("http_body_first_bytes", names)
        self.assertNotIn("json_result", names)
        self.assertNotIn("response", names)

    def test_trickle_response_hits_total_deadline_despite_read_activity(self):
        self.mode = "trickle"
        started = time.monotonic()
        with self.assertRaises(config.ChatJSONError) as raised:
            self.invoke(deadline=1.5, read_timeout=0.3)
        self.assertLess(time.monotonic() - started, 3)
        self.assertEqual(raised.exception.kind, "total_deadline")
        self.assertTrue(self.disconnected.wait(1))
        self.assertEqual(len(self.requests), 1)

    def test_socket_read_timeout_closes_client_and_does_not_retry(self):
        self.mode = "slow_body"
        with self.assertRaises(config.ChatJSONError) as raised:
            self.invoke(deadline=2, read_timeout=0.25)
        self.assertFalse(config.is_retryable_chat_error(raised.exception.__cause__))
        self.assertTrue(self.disconnected.wait(1))
        self.assertEqual(len(self.requests), 1)
        self.assertTrue(next(row for row in self.records() if row["event"] == "request_client_closed")["local_cleanup_confirmed"])

    def test_tracer_keeps_deadline_contract_for_agent(self):
        self.mode = "slow_headers"
        transport = original_run_transport("glm-5.3-flash", "low", 2)
        transport["profiles"]["low"]["deadline_seconds"] = 1.5
        tracer = Tracer(SimpleNamespace(dir=Path(self.temp.name)))
        result = tracer.chat_json("world.structure", [], transport=transport,
                                  model="glm-5.3-flash", retries=5, max_tokens=32768)
        metadata = result["__error_metadata__"]
        self.assertIn("__error__", result)
        self.assertEqual(metadata["kind"], "total_deadline")
        self.assertEqual(metadata["usage_status"], "unknown")
        self.assertEqual(metadata["token_cap"], 32768)
        self.assertTrue(metadata["local_cleanup_confirmed"])
        self.assertEqual(metadata["remote_cancellation"], "unknown")
        self.assertFalse(metadata["retryable"])
        self.assertEqual(len(metadata["call_id"]), 32)

    def test_client_constructor_failure_still_closes_local_http_client(self):
        clients = []
        original = config.httpx.AsyncClient
        def create(**kw):
            client = original(**kw); clients.append(client); return client
        with patch.object(config.httpx, "AsyncClient", side_effect=create), patch.object(
                config._openai, "AsyncOpenAI", side_effect=ValueError("offline bad SDK configuration")):
            with self.assertRaises(config.ChatJSONError):
                self.invoke()
        self.assertEqual(len(clients), 1)
        self.assertTrue(clients[0].is_closed)
        self.assertFalse(self.requests)


if __name__ == "__main__":
    unittest.main(verbosity=2)
