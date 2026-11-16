"""Shared fixtures: isolated LocalFileStore per test."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_BOOT_STORE = tempfile.mkdtemp(prefix="al_boot_")
os.environ.setdefault("AGENT_LEARNING_STORE_BACKEND", "local")
os.environ.setdefault("AGENT_LEARNING_LOCAL_STORE_DIR", _BOOT_STORE)


@pytest.fixture
def temp_store_dir(monkeypatch, tmp_path):
    d = str(tmp_path / "store")
    os.makedirs(d, exist_ok=True)
    monkeypatch.setenv("AGENT_LEARNING_STORE_BACKEND", "local")
    monkeypatch.setenv("AGENT_LEARNING_LOCAL_STORE_DIR", d)
    return d


@pytest.fixture
def store(temp_store_dir):
    from agent_learning.storage.local import LocalFileStore

    return LocalFileStore(root_dir=temp_store_dir)
