#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def run(cmd: list[str], *, cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if check and result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}\n{stderr}")
    return result


def repo_root() -> Path:
    out = run(["git", "rev-parse", "--show-toplevel"], cwd=Path.cwd()).stdout.strip()
    return Path(out)


def _remote_branch_exists(root: Path, branch: str) -> bool:
    result = run(
        ["git", "ls-remote", "--heads", "origin", branch],
        cwd=root,
        check=False,
    )
    return bool(result.stdout.strip())


def ensure_git_identity(root: Path) -> None:
    name = run(["git", "config", "--get", "user.name"], cwd=root, check=False).stdout.strip()
    email = run(
        ["git", "config", "--get", "user.email"], cwd=root, check=False
    ).stdout.strip()
    if not name:
        run(["git", "config", "user.name", "codex-bot"], cwd=root, check=False)
    if not email:
        run(["git", "config", "user.email", "codex-bot@local"], cwd=root, check=False)


def _cleanup_worktree_files(worktree_dir: Path) -> None:
    for child in list(worktree_dir.iterdir()):
        if child.name == ".git":
            continue
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            try:
                child.unlink()
            except Exception:
                pass


def ensure_state_worktree(
    *, root: Path, worktree_dir: Path, branch: str = "scientist-state"
) -> Path:
    worktree_dir = Path(worktree_dir)
    if not worktree_dir.is_absolute():
        worktree_dir = (root / worktree_dir).resolve()
    worktree_dir.parent.mkdir(parents=True, exist_ok=True)

    # If already present, just return it.
    if worktree_dir.exists() and (worktree_dir / ".git").exists():
        return worktree_dir

    if _remote_branch_exists(root, branch):
        run(["git", "fetch", "origin", branch], cwd=root, check=False)
        run(["git", "worktree", "add", str(worktree_dir), branch], cwd=root)
        ensure_git_identity(worktree_dir)
        return worktree_dir

    # Create new orphan state branch in an isolated worktree.
    run(["git", "worktree", "add", str(worktree_dir), "--detach"], cwd=root)
    ensure_git_identity(worktree_dir)

    run(["git", "checkout", "--orphan", branch], cwd=worktree_dir)
    # Remove tracked files from the worktree index/tree.
    run(["git", "rm", "-rf", "."], cwd=worktree_dir, check=False)
    _cleanup_worktree_files(worktree_dir)

    # Minimal branch content.
    (worktree_dir / "state").mkdir(parents=True, exist_ok=True)
    (worktree_dir / "README.md").write_text(
        "\n".join(
            [
                "# scientist-state",
                "",
                "Append-only state/ledger branch for scientist loop automation.",
                "Do not merge into main.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (worktree_dir / "state" / ".keep").write_text("", encoding="utf-8")

    run(["git", "add", "-A"], cwd=worktree_dir)
    run(["git", "commit", "-m", f"chore: initialize {branch}"], cwd=worktree_dir)
    run(["git", "push", "origin", f"HEAD:{branch}"], cwd=worktree_dir)
    return worktree_dir


def main() -> int:
    root = repo_root()
    worktree_dir_raw = (os.environ.get("KEIBA_STATE_REPO_DIR") or "").strip()
    if not worktree_dir_raw:
        raise RuntimeError("Set KEIBA_STATE_REPO_DIR to create/use the state worktree.")
    worktree_dir = Path(worktree_dir_raw)
    ensure_state_worktree(root=root, worktree_dir=worktree_dir)
    print(str(worktree_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
