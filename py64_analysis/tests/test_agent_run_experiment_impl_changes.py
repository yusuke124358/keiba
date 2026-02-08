from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest


def _load_module():
    root = Path(__file__).resolve().parents[2]
    module_path = root / "scripts" / "agent" / "run_experiment.py"
    spec = importlib.util.spec_from_file_location("run_experiment", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


def _run(cwd: Path, *args: str) -> str:
    proc = subprocess.run(
        list(args),
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return (proc.stdout or "").strip()


def _init_git_repo(tmp_path: Path) -> str:
    _run(tmp_path, "git", "init")
    _run(tmp_path, "git", "config", "user.email", "test@example.com")
    _run(tmp_path, "git", "config", "user.name", "Test")
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    (tmp_path / "config" / "a.txt").write_text("a\n", encoding="utf-8")
    _run(tmp_path, "git", "add", "-A")
    _run(tmp_path, "git", "commit", "-m", "init")
    return _run(tmp_path, "git", "rev-parse", "HEAD")


def test_collect_impl_changes_detects_worktree_changes(tmp_path: Path):
    base = _init_git_repo(tmp_path)
    (tmp_path / "config" / "a.txt").write_text("b\n", encoding="utf-8")

    paths, head, mode = MODULE.collect_impl_changes(root=tmp_path, base_commit=base)
    assert mode == "worktree"
    assert head == base
    assert "config/a.txt" in paths


def test_collect_impl_changes_detects_committed_changes(tmp_path: Path):
    base = _init_git_repo(tmp_path)
    (tmp_path / "config" / "a.txt").write_text("c\n", encoding="utf-8")
    _run(tmp_path, "git", "add", "-A")
    _run(tmp_path, "git", "commit", "-m", "change")
    head = _run(tmp_path, "git", "rev-parse", "HEAD")

    paths, head_after, mode = MODULE.collect_impl_changes(
        root=tmp_path, base_commit=base
    )
    assert mode == "commits"
    assert head_after == head
    assert "config/a.txt" in paths


def test_collect_impl_changes_raises_when_no_changes(tmp_path: Path):
    base = _init_git_repo(tmp_path)
    with pytest.raises(RuntimeError):
        MODULE.collect_impl_changes(root=tmp_path, base_commit=base)

