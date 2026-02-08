#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path


def run(
    cmd: list[str], *, cwd: Path, check: bool = True
) -> subprocess.CompletedProcess:
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if check and result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\n{stderr}"
        )
    return result


def repo_root() -> Path:
    out = run(["git", "rev-parse", "--show-toplevel"], cwd=Path.cwd()).stdout.strip()
    return Path(out)


def _origin_url(root: Path) -> str:
    url = run(["git", "remote", "get-url", "origin"], cwd=root).stdout.strip()
    if not url:
        raise RuntimeError(
            "Unable to determine origin URL (git remote get-url origin)."
        )
    return url


def _remote_branch_exists(root: Path, branch: str) -> bool:
    result = run(
        ["git", "ls-remote", "--heads", "origin", branch],
        cwd=root,
        check=False,
    )
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(
            f"Unable to query remote origin for branch {branch!r}.\n{stderr}"
        )
    return bool(result.stdout.strip())


def ensure_git_identity(root: Path) -> None:
    name = run(
        ["git", "config", "--get", "user.name"], cwd=root, check=False
    ).stdout.strip()
    email = run(
        ["git", "config", "--get", "user.email"], cwd=root, check=False
    ).stdout.strip()
    if not name:
        run(["git", "config", "user.name", "codex-bot"], cwd=root, check=False)
    if not email:
        run(["git", "config", "user.email", "codex-bot@local"], cwd=root, check=False)


def _is_git_repo(path: Path) -> bool:
    if not path.exists():
        return False
    result = run(
        ["git", "rev-parse", "--is-inside-work-tree"],
        cwd=path,
        check=False,
    )
    return result.returncode == 0 and result.stdout.strip() == "true"


def _backup_broken_dir(path: Path) -> None:
    if not path.exists():
        return
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    backup = path.with_name(f"{path.name}.broken_{stamp}")
    try:
        path.rename(backup)
    except Exception:
        shutil.rmtree(path, ignore_errors=True)


def _init_state_branch(*, repo_dir: Path, branch: str) -> None:
    # Minimal branch content.
    (repo_dir / "state").mkdir(parents=True, exist_ok=True)
    (repo_dir / "README.md").write_text(
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
    (repo_dir / "state" / ".keep").write_text("", encoding="utf-8")


def ensure_state_worktree(
    *, root: Path, worktree_dir: Path, branch: str = "scientist-state"
) -> Path:
    worktree_dir = Path(worktree_dir)
    if not worktree_dir.is_absolute():
        worktree_dir = (root / worktree_dir).resolve()
    worktree_dir.parent.mkdir(parents=True, exist_ok=True)

    # NOTE: We intentionally do NOT use `git worktree` here.
    # Persisting a worktree across GitHub Actions jobs is brittle because the worktree's
    # `.git` file points at `.../.git/worktrees/<name>` inside the *job checkout*.
    # When that checkout changes or is cleaned, the worktree becomes unusable.
    #
    # Instead, we keep an independent lightweight repo checkout (just the state branch)
    # under `$KEIBA_STATE_REPO_DIR` and fetch/rebase/push normally.

    origin_url = _origin_url(root)

    if _is_git_repo(worktree_dir):
        # Ensure the state repo can push (workflows set origin with a token).
        run(
            ["git", "remote", "set-url", "origin", origin_url],
            cwd=worktree_dir,
            check=False,
        )
        return worktree_dir

    # Existing directory but not a usable git repo (often a broken old worktree). Move aside.
    if worktree_dir.exists():
        _backup_broken_dir(worktree_dir)
    worktree_dir.mkdir(parents=True, exist_ok=True)

    run(["git", "init"], cwd=worktree_dir)
    run(["git", "remote", "add", "origin", origin_url], cwd=worktree_dir, check=False)
    run(
        ["git", "remote", "set-url", "origin", origin_url],
        cwd=worktree_dir,
        check=False,
    )
    ensure_git_identity(worktree_dir)

    if _remote_branch_exists(worktree_dir, branch):
        run(["git", "fetch", "origin", branch], cwd=worktree_dir)
        run(["git", "checkout", "-B", branch, f"origin/{branch}"], cwd=worktree_dir)
        return worktree_dir

    # Initialize the branch on remote if it doesn't exist yet.
    run(["git", "checkout", "--orphan", branch], cwd=worktree_dir)
    _init_state_branch(repo_dir=worktree_dir, branch=branch)
    run(["git", "add", "-A"], cwd=worktree_dir)
    run(["git", "commit", "-m", f"chore: initialize {branch}"], cwd=worktree_dir)
    run(["git", "push", "origin", f"HEAD:{branch}"], cwd=worktree_dir)
    return worktree_dir


def main() -> int:
    root = repo_root()
    worktree_dir_raw = (os.environ.get("KEIBA_STATE_REPO_DIR") or "").strip()
    if not worktree_dir_raw:
        raise RuntimeError("Set KEIBA_STATE_REPO_DIR to create/use the state repo.")
    worktree_dir = Path(worktree_dir_raw)
    ensure_state_worktree(root=root, worktree_dir=worktree_dir)
    print(str(worktree_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
