"""WordPress draft client. Refuses every host outside loopback."""

from __future__ import annotations

import base64
import json
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

LOOPBACK = {"127.0.0.1", "localhost", "::1"}


class GuardError(RuntimeError):
    pass


class WPDraftClient:
    def __init__(
        self,
        base_url: str,
        user: str,
        app_password: str,
        retries: int = 3,
        timeout: float = 5.0,
    ):
        host = urlparse(base_url).hostname
        if host not in LOOPBACK:
            raise GuardError(f"host {host!r} is niet loopback; harness weigert")
        self.base_url = base_url.rstrip("/")
        token = base64.b64encode(f"{user}:{app_password}".encode()).decode()
        self.auth = f"Basic {token}"
        self.retries = retries
        self.timeout = timeout

    def create_draft(self, title: str, content: str, meta: dict | None = None) -> dict:
        payload = {"title": title, "content": content, "status": "draft"}
        if meta:
            payload["meta"] = meta
        body = json.dumps(payload).encode()
        last_err: int | str | None = None
        for attempt in range(1, self.retries + 1):
            req = Request(
                f"{self.base_url}/wp-json/wp/v2/posts",
                data=body,
                method="POST",
                headers={"Content-Type": "application/json", "Authorization": self.auth},
            )
            try:
                with urlopen(req, timeout=self.timeout) as resp:  # nosec B310
                    return {
                        "ok": True,
                        "status": resp.status,
                        "data": json.loads(resp.read()),
                        "attempts": attempt,
                    }
            except HTTPError as exc:
                last_err = exc.code
                if exc.code < 500:
                    return {"ok": False, "status": exc.code, "attempts": attempt}
            except URLError:
                last_err = "network"
        return {"ok": False, "status": last_err, "attempts": self.retries}
