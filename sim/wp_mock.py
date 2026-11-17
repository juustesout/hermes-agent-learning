"""Loopback mock of WordPress REST POST /wp-json/wp/v2/posts.

Accepts only status=draft. Publish is 403. Bound to 127.0.0.1/localhost.
"""

from __future__ import annotations

import base64
import binascii
import json
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

LOOPBACK = {"127.0.0.1", "localhost", "::1"}


class WPMock:
    def __init__(
        self,
        user="crashtest",
        app_password="secret",
        error_rate=0.0,
        latency_s=0.0,
        seed=42,
        fail_codes=(500, 503),
    ):
        self.user = user
        self.app_password = app_password
        self.error_rate = error_rate
        self.latency_s = latency_s
        self.fail_codes = tuple(fail_codes)
        self.rng = random.Random(seed)
        self.posts: list[dict[str, Any]] = []
        self.rejected: list[dict[str, Any]] = []
        self.lock = threading.Lock()
        self._next_id = 1000
        self.server: ThreadingHTTPServer | None = None

    def _auth_ok(self, header: str | None) -> bool:
        if not header or not header.startswith("Basic "):
            return False
        try:
            raw = base64.b64decode(header[6:]).decode("utf-8")
        except (ValueError, binascii.Error, UnicodeDecodeError):
            return False
        return raw == f"{self.user}:{self.app_password}"

    def handle_create(self, body: dict[str, Any], auth: str | None) -> tuple[int, dict]:
        if self.latency_s:
            time.sleep(self.latency_s)
        with self.lock:
            if not self._auth_ok(auth):
                self.rejected.append({"reason": "auth", "body": body})
                return 401, {"code": "rest_cannot_create", "message": "Auth mislukt"}
            if self.error_rate and self.rng.random() < self.error_rate:
                code = self.rng.choice(self.fail_codes)
                self.rejected.append({"reason": "injected", "code": code})
                return code, {"code": "injected_error", "message": "Gesimuleerde storing"}
            if body.get("status") != "draft":
                self.rejected.append({"reason": "not_draft", "body": body})
                return 403, {
                    "code": "rest_forbidden",
                    "message": "Alleen drafts zijn toegestaan in deze mock",
                }
            if not body.get("title"):
                return 400, {"code": "rest_invalid_param", "message": "title ontbreekt"}
            self._next_id += 1
            post = {
                "id": self._next_id,
                "status": "draft",
                "title": body["title"],
                "content": body.get("content", ""),
                "meta": body.get("meta", {}),
            }
            self.posts.append(post)
            return 201, post

    def start(self, host: str = "127.0.0.1", port: int = 0) -> str:
        if host not in LOOPBACK:
            raise ValueError("WPMock draait alleen op loopback")
        mock = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_POST(self):
                if self.path.rstrip("/") != "/wp-json/wp/v2/posts":
                    return self._send(404, {"code": "rest_no_route"})
                length = int(self.headers.get("Content-Length", 0))
                try:
                    body = json.loads(self.rfile.read(length) or b"{}")
                except json.JSONDecodeError:
                    return self._send(400, {"code": "rest_invalid_json"})
                code, payload = mock.handle_create(body, self.headers.get("Authorization"))
                self._send(code, payload)

            def do_GET(self):
                if self.path.rstrip("/") == "/__stats":
                    with mock.lock:
                        return self._send(
                            200,
                            {"posts": len(mock.posts), "rejected": len(mock.rejected)},
                        )
                self._send(404, {"code": "rest_no_route"})

            def _send(self, code, payload):
                data = json.dumps(payload).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer((host, port), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return f"http://{host}:{self.server.server_address[1]}"

    def stop(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
