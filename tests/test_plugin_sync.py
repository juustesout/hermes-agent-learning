from __future__ import annotations

from pathlib import Path

import plugin_sync


def _mini_repo(root: Path) -> Path:
    repo = root / "repo"
    repo.mkdir()
    (repo / "plugin.yaml").write_text("name: agent-learning\n", encoding="utf-8")
    (repo / "hermes_plugin").mkdir()
    (repo / "hermes_plugin" / "__init__.py").write_text("# plugin\n", encoding="utf-8")
    (repo / "features.py").write_text("# features\n", encoding="utf-8")
    (repo / "opslog.py").write_text("# opslog\n", encoding="utf-8")
    return repo


def test_install_links_whole_repo(tmp_path):
    repo = _mini_repo(tmp_path)
    dest = tmp_path / "plugins" / "agent-learning"
    assert plugin_sync.install_plugin(repo=repo, dest=dest) == 0
    assert dest.is_symlink()
    assert dest.resolve() == repo.resolve()
    assert (dest / "features.py").is_file()
    assert (dest / "opslog.py").is_file()
    assert plugin_sync.check_plugin_sync(repo=repo, dest=dest) == 0


def test_install_refuses_existing_path(tmp_path):
    repo = _mini_repo(tmp_path)
    dest = tmp_path / "plugins" / "agent-learning"
    dest.parent.mkdir(parents=True)
    dest.mkdir()
    (dest / "plugin.yaml").write_text("name: leftover\n", encoding="utf-8")
    assert plugin_sync.install_plugin(repo=repo, dest=dest) == 1
    assert dest.is_dir()
    assert not dest.is_symlink()


def test_check_fails_on_subset_copy(tmp_path):
    repo = _mini_repo(tmp_path)
    dest = tmp_path / "plugins" / "agent-learning"
    dest.mkdir(parents=True)
    (dest / "plugin.yaml").write_text((repo / "plugin.yaml").read_text(), encoding="utf-8")
    (dest / "hermes_plugin").mkdir()
    (dest / "hermes_plugin" / "__init__.py").write_text(
        (repo / "hermes_plugin" / "__init__.py").read_text(), encoding="utf-8"
    )
    assert plugin_sync.check_plugin_sync(repo=repo, dest=dest) == 1


def test_check_fails_when_dest_missing(tmp_path):
    repo = _mini_repo(tmp_path)
    dest = tmp_path / "missing"
    assert plugin_sync.check_plugin_sync(repo=repo, dest=dest) == 1
