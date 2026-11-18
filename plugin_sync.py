"""Install the whole repo as the Hermes plugin and check it matches."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

DIFF_EXCLUDE = (".git", "store", "dashboard", "__pycache__", ".venv", "venv", ".pytest_cache")


def repo_root() -> Path:
    return Path(__file__).resolve().parent


def default_dest() -> Path:
    env = os.environ.get("HERMES_PLUGIN_DIR", "").strip()
    if env:
        return Path(env)
    return Path.home() / ".hermes" / "plugins" / "agent-learning"


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return False


def install_plugin(*, repo: Path, dest: Path) -> int:
    repo = repo.resolve()
    dest = dest.expanduser()
    if not (repo / "plugin.yaml").is_file() or not (repo / "hermes_plugin").is_dir():
        print(f"not a plugin repo: {repo}", file=sys.stderr)
        return 1
    if dest.exists() or dest.is_symlink():
        if dest.is_symlink() and _same_path(dest, repo):
            print(f"linked {dest} -> {repo}")
            return 0
        print(
            f"refusing to replace existing path: {dest}\n"
            f"point HERMES_PLUGIN_DIR at an empty location or remove the path yourself",
            file=sys.stderr,
        )
        return 1
    parent = dest.parent
    parent.mkdir(parents=True, exist_ok=True)
    os.symlink(repo, dest)
    print(f"linked {dest} -> {repo}")
    return 0


def check_plugin_sync(*, repo: Path, dest: Path) -> int:
    repo = repo.resolve()
    dest = dest.expanduser()
    if not dest.exists() and not dest.is_symlink():
        print(f"plugin dest missing: {dest}", file=sys.stderr)
        return 1
    if dest.is_symlink() and _same_path(dest, repo):
        print(f"ok symlink {dest} -> {repo}")
        return 0
    cmd = ["diff", "-rq"]
    for name in DIFF_EXCLUDE:
        cmd.extend(["-x", name])
    cmd.extend([str(dest), str(repo)])
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0 and not result.stdout.strip() and not result.stderr.strip():
        print(f"ok copy {dest} matches {repo}")
        return 0
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    print("plugin dest differs from repo", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="plugin_sync")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("install", "check"):
        p = sub.add_parser(name)
        p.add_argument("--repo", default=str(repo_root()))
        p.add_argument("--dest", default="")
    args = parser.parse_args(argv)
    repo = Path(args.repo)
    dest = Path(args.dest) if args.dest else default_dest()
    if args.cmd == "install":
        return install_plugin(repo=repo, dest=dest)
    return check_plugin_sync(repo=repo, dest=dest)


if __name__ == "__main__":
    raise SystemExit(main())
