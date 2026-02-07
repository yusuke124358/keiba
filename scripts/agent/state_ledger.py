#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from json_canonical import canonical_dumps, sha256_hex
from state_worktree import ensure_git_identity, ensure_state_worktree, run


STATE_BRANCH = "scientist-state"


class LeaseAlreadyHeldError(RuntimeError):
    pass


def _ensure_existing_lease_is_idempotent(
    *, existing: dict[str, Any], new_event: dict[str, Any], out_path: Path
) -> None:
    """
    When a lease event file already exists, only allow idempotent re-append by the same owner/run_id.
    """
    existing_owner = str(existing.get("owner") or "").strip()
    existing_run_id = str(existing.get("run_id") or "").strip()
    new_owner = str(new_event.get("owner") or "").strip()
    new_run_id = str(new_event.get("run_id") or "").strip()

    if existing_owner == new_owner and existing_run_id == new_run_id:
        return
    raise LeaseAlreadyHeldError(
        "Lease already held by another owner/run_id. "
        f"path={out_path} existing_owner={existing_owner!r} existing_run_id={existing_run_id!r} "
        f"new_owner={new_owner!r} new_run_id={new_run_id!r}"
    )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def compute_event_id(identity: dict[str, Any]) -> str:
    payload = canonical_dumps(identity)
    return sha256_hex(payload)


def validate_event(event: dict[str, Any], schema_path: Path) -> None:
    try:
        import jsonschema  # type: ignore[import-untyped]
    except Exception as exc:
        raise RuntimeError(
            "jsonschema is required for scientist-state event validation. "
            "Install with: pip install jsonschema"
        ) from exc
    schema = json.loads(schema_path.read_text(encoding="utf-8-sig"))
    jsonschema.validate(instance=event, schema=schema)


def state_events_dir(campaign_id: str) -> Path:
    return Path("state") / "campaigns" / campaign_id / "events"


def state_patches_dir(campaign_id: str, seed_id: str) -> Path:
    return Path("state") / "campaigns" / campaign_id / "patches" / seed_id


def git_sync_state_branch(state_root: Path, *, retries: int = 5) -> None:
    """
    Ensure we're on top of the latest `scientist-state` tip before committing.

    Conflicts should be rare because we only append new files, but pushes can still race.
    We treat any rebase failure as transient and recover by aborting and hard-resetting
    to the remote tip.
    """
    delay = 1.0
    last_err = ""
    for _ in range(max(int(retries), 1)):
        run(["git", "fetch", "origin", STATE_BRANCH], cwd=state_root, check=False)
        rebase = run(
            ["git", "rebase", f"origin/{STATE_BRANCH}"],
            cwd=state_root,
            check=False,
        )
        if rebase.returncode == 0:
            return
        last_err = (rebase.stderr or "").strip()
        run(["git", "rebase", "--abort"], cwd=state_root, check=False)
        run(
            ["git", "reset", "--hard", f"origin/{STATE_BRANCH}"],
            cwd=state_root,
            check=False,
        )
        time.sleep(delay)
        delay = min(delay * 2.0, 10.0)
    raise RuntimeError(f"Unable to sync state branch after retries.\n{last_err}")


def git_push_state_branch(state_root: Path, *, retries: int = 6) -> None:
    delay = 1.0
    for attempt in range(max(retries, 1)):
        result = run(
            ["git", "push", "origin", f"HEAD:{STATE_BRANCH}"],
            cwd=state_root,
            check=False,
        )
        if result.returncode == 0:
            return
        # Non-fast-forward or transient failure: rebase and retry.
        run(["git", "fetch", "origin", STATE_BRANCH], cwd=state_root, check=False)
        run(["git", "rebase", f"origin/{STATE_BRANCH}"], cwd=state_root, check=False)
        time.sleep(delay)
        delay = min(delay * 2.0, 15.0)
    stderr = (
        (result.stderr or "").strip()
        if isinstance(result, subprocess.CompletedProcess)
        else ""
    )
    raise RuntimeError(f"Failed to push state branch after retries.\n{stderr}")


def append_event(
    *,
    repo_root: Path,
    state_repo_dir: Path,
    campaign_id: str,
    event: dict[str, Any],
    schema_path: Path,
    commit_message: str,
) -> Path:
    """
    Append an event to the state branch as a new file. Idempotent by event_id filename.
    Returns the event file path (within the state worktree).
    """
    state_repo_dir = ensure_state_worktree(root=repo_root, worktree_dir=state_repo_dir)
    ensure_git_identity(state_repo_dir)

    # Ensure we are on the correct branch.
    run(["git", "checkout", STATE_BRANCH], cwd=state_repo_dir)
    git_sync_state_branch(state_repo_dir)

    validate_event(event, schema_path)
    event_id = str(event.get("event_id") or "").strip()
    if not event_id:
        raise RuntimeError("event_id is required.")

    out_dir = state_repo_dir / state_events_dir(campaign_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{event_id}.json"
    if out_path.exists():
        event_type = str(event.get("event_type") or "").strip()
        if event_type == "lease_acquired":
            try:
                existing = json.loads(out_path.read_text(encoding="utf-8-sig"))
            except Exception:
                raise LeaseAlreadyHeldError(
                    f"Lease already held (unable to read existing lease file): {out_path}"
                )
            _ensure_existing_lease_is_idempotent(
                existing=existing, new_event=event, out_path=out_path
            )
        return out_path

    out_path.write_text(
        json.dumps(event, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    run(["git", "add", str(out_path)], cwd=state_repo_dir)
    run(["git", "commit", "-m", commit_message], cwd=state_repo_dir)
    git_push_state_branch(state_repo_dir)
    return out_path


def store_patch(
    *,
    repo_root: Path,
    state_repo_dir: Path,
    campaign_id: str,
    seed_id: str,
    patch_sha256: str,
    patch_src: Path,
    commit_message: str,
) -> Path:
    state_repo_dir = ensure_state_worktree(root=repo_root, worktree_dir=state_repo_dir)
    ensure_git_identity(state_repo_dir)
    run(["git", "checkout", STATE_BRANCH], cwd=state_repo_dir)
    git_sync_state_branch(state_repo_dir)

    if not patch_src.exists():
        raise RuntimeError(f"Patch source not found: {patch_src}")

    out_dir = state_repo_dir / state_patches_dir(campaign_id, seed_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{patch_sha256}.diff"
    if out_path.exists():
        return out_path

    out_path.write_bytes(patch_src.read_bytes())
    run(["git", "add", str(out_path)], cwd=state_repo_dir)
    run(["git", "commit", "-m", commit_message], cwd=state_repo_dir)
    git_push_state_branch(state_repo_dir)
    return out_path


def load_events(*, state_repo_dir: Path, campaign_id: str) -> list[dict[str, Any]]:
    events_dir = state_repo_dir / state_events_dir(campaign_id)
    if not events_dir.exists():
        return []
    out: list[dict[str, Any]] = []
    for p in sorted(events_dir.glob("*.json")):
        try:
            out.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return out


def owner_string() -> str:
    parts = []
    for k in ("RUNNER_NAME", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT"):
        v = (os.environ.get(k) or "").strip()
        if v:
            parts.append(f"{k}={v}")
    return " ".join(parts) if parts else "local"
