"""Mechanical tests: loopback WP mock, drafts only, retries, no remote hosts."""

from __future__ import annotations

import base64
import json
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from sim.client import GuardError, WPDraftClient
from sim.wp_mock import WPMock


@pytest.fixture
def server():
    mock = WPMock(error_rate=0.0)
    base = mock.start()
    yield mock, base
    mock.stop()


def make_client(base):
    return WPDraftClient(base, "crashtest", "secret", retries=3)


def test_guard_refuses_remote_host():
    with pytest.raises(GuardError):
        WPDraftClient("https://example.com", "u", "p")


def test_publish_is_rejected(server):
    mock, base = server
    tok = base64.b64encode(b"crashtest:secret").decode()
    req = Request(
        f"{base}/wp-json/wp/v2/posts",
        method="POST",
        data=json.dumps({"title": "x", "status": "publish"}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Basic {tok}"},
    )
    with pytest.raises(HTTPError) as exc:
        urlopen(req, timeout=5)  # nosec B310
    assert exc.value.code == 403
    assert len(mock.posts) == 0


def test_wrong_password_is_401(server):
    mock, base = server
    bad = WPDraftClient(base, "crashtest", "wrong", retries=3)
    res = bad.create_draft("t", "c")
    assert res["ok"] is False and res["status"] == 401
    assert res["attempts"] == 1
    assert len(mock.posts) == 0


def test_300_runs_accelerated(server):
    mock, base = server
    client = make_client(base)
    t0 = time.time()
    results = [client.create_draft(f"run-{i}", "inhoud") for i in range(300)]
    elapsed = time.time() - t0
    assert all(r["ok"] for r in results)
    assert len(mock.posts) == 300
    assert all(p["status"] == "draft" for p in mock.posts)
    assert len({p["id"] for p in mock.posts}) == 300
    assert elapsed < 60


def test_retry_on_injected_5xx():
    mock = WPMock(error_rate=0.5, seed=7)
    base = mock.start()
    try:
        client = make_client(base)
        results = [client.create_draft(f"r{i}", "c") for i in range(200)]
        ok = [r for r in results if r["ok"]]
        assert 0.80 <= len(ok) / len(results) <= 0.95
        assert any(r["attempts"] > 1 for r in results)
        assert len(mock.posts) == len(ok)
    finally:
        mock.stop()
