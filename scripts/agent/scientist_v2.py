#!/usr/bin/env python3
"""
Scientist Loop V2

Key properties:
- Exploration is results-only (no PR per experiment).
- Campaign windows are fixed with non-overlapping Stage1/Stage2/Holdout test periods.
- Central append-only ledger on the `scientist-state` branch is the source of truth.
- Sharding provides safe parallelism; per-shard concurrency should be serialized.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "PyYAML is required to run scientist_v2. Install with: pip install pyyaml"
    ) from exc

from scientist_campaign import CampaignConfig, StageRules, load_campaign_by_id
from shard import assigned_shard, parse_shard
from state_ledger import (
    LeaseAlreadyHeldError,
    append_event,
    compute_event_id,
    load_events,
    owner_string,
    store_patch,
    utc_now_iso,
)
from state_worktree import ensure_state_worktree


def run(
    cmd: list[str],
    *,
    cwd: Path,
    check: bool = True,
    capture_output: bool = True,
    text: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    result = subprocess.run(
        cmd,
        cwd=cwd,
        check=False,
        capture_output=capture_output,
        text=text,
        env=env,
    )
    if check and result.returncode != 0:
        stderr = (result.stderr or "").strip()
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\n{stderr}"
        )
    return result


def repo_root() -> Path:
    out = run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=Path.cwd(),
        capture_output=True,
    ).stdout.strip()
    return Path(out)


def parse_utc(s: str) -> datetime:
    # All ledger timestamps are written in UTC with a Z suffix.
    return datetime.strptime(str(s).strip(), "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc
    )


@dataclass(frozen=True)
class Seed:
    seed_id: str
    title: str
    hypothesis: str
    change_scope: str
    acceptance_criteria: list[str]
    metrics: list[str]
    risk_level: str
    max_diff_size: int


def load_seeds(path: Path) -> list[Seed]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise RuntimeError(f"seed_hypotheses.yaml must be a non-empty list: {path}")
    out: list[Seed] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise RuntimeError(f"seed_hypotheses entry {idx} is not a mapping")
        seed_id = str(item.get("id") or "").strip()
        if not seed_id:
            raise RuntimeError(f"seed_hypotheses entry {idx} missing id")
        out.append(
            Seed(
                seed_id=seed_id,
                title=str(item.get("title") or "").strip(),
                hypothesis=str(item.get("hypothesis") or "").strip(),
                change_scope=str(item.get("change_scope") or "").strip(),
                acceptance_criteria=[
                    str(x) for x in (item.get("acceptance_criteria") or [])
                ],
                metrics=[str(x) for x in (item.get("metrics") or [])],
                risk_level=str(item.get("risk_level") or "medium").strip(),
                max_diff_size=int(item.get("max_diff_size") or 1),
            )
        )
    return out


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_run_id(*, campaign_id: str, stage: str, seed_id: str, attempt: int) -> str:
    # Keep this git-branch-safe (run_experiment uses it in branch names).
    return f"{campaign_id}-{stage}-{seed_id}-a{attempt}"


def build_holdout_command(
    *,
    run_id: str,
    splits: dict[str, str],
    test_start: str,
    test_end: str,
) -> str:
    return (
        "py64_analysis/scripts/run_holdout.py"
        f" --train-start {splits['train_start']} --train-end {splits['train_end']}"
        f" --valid-start {splits['valid_start']} --valid-end {splits['valid_end']}"
        f" --test-start {test_start} --test-end {test_end}"
        f" --name {run_id} --out-dir data/holdout_runs/{run_id}"
    )


def build_plan(
    *,
    seed: Seed,
    run_id: str,
    eval_command: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "seed_id": seed.seed_id,
        "title": seed.title,
        "hypothesis": seed.hypothesis,
        "change_scope": seed.change_scope,
        "acceptance_criteria": seed.acceptance_criteria,
        "metrics": seed.metrics,
        "risk_level": seed.risk_level,
        "max_diff_size": int(seed.max_diff_size),
        "eval_command": [eval_command],
        "metrics_path": f"data/holdout_runs/{run_id}/metrics.json",
        "decision": "do",
        "reason": reason,
    }


def stage_rules(cfg: CampaignConfig, stage: str) -> StageRules:
    if stage == "stage1":
        return cfg.stage1
    if stage == "stage2":
        return cfg.stage2
    if stage == "holdout":
        return cfg.holdout
    raise ValueError(f"Unknown stage: {stage}")


def _null_metrics_summary() -> dict[str, Any]:
    # Schema requires delta_roi_ci95 even when the run failed and no metrics exist.
    return {
        "delta_roi": None,
        "delta_roi_ci95": [0.0, 0.0],
        "p_one_sided_delta_le_0": None,
        "n_bets": None,
        "n_days": None,
        "test_period": "N/A",
        "delta_max_drawdown": None,
    }


def metrics_summary_from_result(result: dict[str, Any]) -> dict[str, Any]:
    try:
        delta_roi = (result.get("deltas") or {}).get("delta_roi")
        delta_dd = (result.get("deltas") or {}).get("delta_max_drawdown")
        stats = result.get("stats") or {}
        ci = stats.get("delta_roi_ci95")
        p = stats.get("p_one_sided_delta_le_0")
        metrics = result.get("metrics") or {}
        n_bets = metrics.get("n_bets")
        n_days = metrics.get("n_days")
        test_period = str(metrics.get("test_period") or "N/A")
        if not (isinstance(ci, list) and len(ci) == 2):
            return _null_metrics_summary()
        return {
            "delta_roi": delta_roi if delta_roi is None else float(delta_roi),
            "delta_roi_ci95": [float(ci[0]), float(ci[1])],
            "p_one_sided_delta_le_0": None if p is None else float(p),
            "n_bets": None if n_bets is None else int(n_bets),
            "n_days": None if n_days is None else int(n_days),
            "test_period": test_period,
            "delta_max_drawdown": delta_dd if delta_dd is None else float(delta_dd),
        }
    except Exception:
        return _null_metrics_summary()


def evaluate_stage_decision(
    *, stage: str, rules: StageRules, result: dict[str, Any]
) -> str:
    """
    Decide candidate/accept/reject/iterate/needs-human for the central ledger.

    Notes:
    - Stage1 uses "candidate" as the positive decision.
    - Stage2/Holdout use "accept" as the positive decision.
    """
    pos = "candidate" if stage == "stage1" else "accept"

    decision_node = result.get("decision") or {}
    leakage = str(decision_node.get("leakage_pre_race_only") or "").strip().lower()
    if leakage == "no":
        return "needs-human"

    ms = metrics_summary_from_result(result)
    n_bets = ms.get("n_bets")
    n_days = ms.get("n_days")
    if n_bets is None or n_days is None:
        return "iterate"
    if int(n_bets) < int(rules.min_bets) or int(n_days) < int(rules.min_days):
        return "iterate"

    ci = ms.get("delta_roi_ci95") or [0.0, 0.0]
    lo = float(ci[0])
    hi = float(ci[1])
    if hi < 0.0:
        return "reject"

    p = ms.get("p_one_sided_delta_le_0")
    if p is None:
        return "iterate"

    if lo > float(rules.delta_roi_ci95_lower_gt) and float(p) <= float(
        rules.p_one_sided_delta_le_0_max
    ):
        return pos
    return "iterate"


def event_id_for(identity: dict[str, Any]) -> str:
    return compute_event_id(identity)


def canonical_owner() -> str:
    # Avoid including volatile fields in event_id identity (owner is stored, not hashed).
    return owner_string()


@dataclass(frozen=True)
class Lease:
    seed_id: str
    stage: str
    attempt: int
    period_key: str
    run_id: str
    owner: str
    base_commit: str
    started_at: datetime


@dataclass(frozen=True)
class Finished:
    seed_id: str
    stage: str
    attempt: int
    period_key: str
    run_id: str
    owner: str
    result: str
    decision: str
    finished_at: datetime
    patch_ref: dict[str, str] | None
    metrics_summary: dict[str, Any]
    last_error: str


@dataclass(frozen=True)
class Heartbeat:
    seed_id: str
    stage: str
    attempt: int
    heartbeat_bucket: int
    last_heartbeat_at: datetime


@dataclass(frozen=True)
class Reconciled:
    seed_id: str
    stage: str
    attempt: int
    reconcile_action: str
    reconciled_at: datetime
    reason: str


class LedgerIndex:
    def __init__(self, events: list[dict[str, Any]]):
        self.leases: dict[tuple[str, str, int], Lease] = {}
        self.finished: dict[tuple[str, str, int], Finished] = {}
        self.heartbeats: dict[tuple[str, str, int], Heartbeat] = {}
        self.reconciled: dict[tuple[str, str, int], Reconciled] = {}
        self._build(events)

    def _build(self, events: list[dict[str, Any]]) -> None:
        for ev in events:
            if not isinstance(ev, dict):
                continue
            et = str(ev.get("event_type") or "").strip()
            seed_id = str(ev.get("seed_id") or "").strip()
            stage = str(ev.get("stage") or "").strip()
            if not seed_id or not stage:
                continue

            if et == "lease_acquired":
                key = (seed_id, stage, int(ev.get("attempt") or 0))
                try:
                    self.leases[key] = Lease(
                        seed_id=seed_id,
                        stage=stage,
                        attempt=int(ev["attempt"]),
                        period_key=str(ev["period_key"]),
                        run_id=str(ev["run_id"]),
                        owner=str(ev.get("owner") or ""),
                        base_commit=str(ev["base_commit"]),
                        started_at=parse_utc(str(ev["started_at"])),
                    )
                except Exception:
                    continue

            elif et == "finished":
                key = (seed_id, stage, int(ev.get("attempt") or 0))
                try:
                    patch_ref = ev.get("patch_ref")
                    if not isinstance(patch_ref, dict):
                        patch_ref = None
                    self.finished[key] = Finished(
                        seed_id=seed_id,
                        stage=stage,
                        attempt=int(ev["attempt"]),
                        period_key=str(ev["period_key"]),
                        run_id=str(ev["run_id"]),
                        owner=str(ev.get("owner") or ""),
                        result=str(ev.get("result") or ""),
                        decision=str(ev.get("decision") or ""),
                        finished_at=parse_utc(str(ev["finished_at"])),
                        patch_ref=patch_ref,
                        metrics_summary=dict(ev.get("metrics_summary") or {}),
                        last_error=str(ev.get("last_error") or ""),
                    )
                except Exception:
                    continue

            elif et == "heartbeat":
                key = (seed_id, stage, int(ev.get("attempt") or 0))
                try:
                    hb = Heartbeat(
                        seed_id=seed_id,
                        stage=stage,
                        attempt=int(ev["attempt"]),
                        heartbeat_bucket=int(ev["heartbeat_bucket"]),
                        last_heartbeat_at=parse_utc(str(ev["last_heartbeat_at"])),
                    )
                    prev = self.heartbeats.get(key)
                    if prev is None or hb.last_heartbeat_at > prev.last_heartbeat_at:
                        self.heartbeats[key] = hb
                except Exception:
                    continue

            elif et == "reconciled":
                key = (seed_id, stage, int(ev.get("attempt") or 0))
                try:
                    rec = Reconciled(
                        seed_id=seed_id,
                        stage=stage,
                        attempt=int(ev["attempt"]),
                        reconcile_action=str(ev["reconcile_action"]),
                        reconciled_at=parse_utc(str(ev["reconciled_at"])),
                        reason=str(ev.get("reason") or ""),
                    )
                    prev = self.reconciled.get(key)
                    if prev is None or rec.reconciled_at > prev.reconciled_at:
                        self.reconciled[key] = rec
                except Exception:
                    continue

    def max_attempt(self, seed_id: str, stage: str) -> int:
        m = 0
        for sid, st, att in self.leases.keys():
            if sid == seed_id and st == stage:
                m = max(m, att)
        for sid, st, att in self.finished.keys():
            if sid == seed_id and st == stage:
                m = max(m, att)
        for sid, st, att in self.reconciled.keys():
            if sid == seed_id and st == stage:
                m = max(m, att)
        return m

    def next_attempt(self, seed_id: str, stage: str) -> int:
        return self.max_attempt(seed_id, stage) + 1

    def has_success(self, seed_id: str, stage: str) -> bool:
        for (sid, st, _), fin in self.finished.items():
            if sid == seed_id and st == stage and fin.result == "success":
                return True
        return False

    def latest_success(
        self, seed_id: str, stage: str, *, decision: str | None = None
    ) -> Finished | None:
        best: Finished | None = None
        for (sid, st, _), fin in self.finished.items():
            if sid != seed_id or st != stage:
                continue
            if fin.result != "success":
                continue
            if decision and fin.decision != decision:
                continue
            if best is None or fin.finished_at > best.finished_at:
                best = fin
        return best

    def active_lease(self, seed_id: str, stage: str) -> Lease | None:
        active: Lease | None = None
        for (sid, st, att), lease in self.leases.items():
            if sid != seed_id or st != stage:
                continue
            key = (sid, st, att)
            if key in self.finished:
                continue
            if key in self.reconciled:
                continue
            if active is None or lease.started_at > active.started_at:
                active = lease
        return active

    def last_activity(self, lease: Lease) -> datetime:
        key = (lease.seed_id, lease.stage, lease.attempt)
        hb = self.heartbeats.get(key)
        if hb is None:
            return lease.started_at
        return max(lease.started_at, hb.last_heartbeat_at)

    def lease_for_run(self, run_id: str) -> Lease | None:
        for lease in self.leases.values():
            if lease.run_id == run_id:
                return lease
        return None


def ensure_git_identity(root: Path) -> None:
    name = run(
        ["git", "config", "--get", "user.name"], cwd=root, check=False
    ).stdout.strip()
    email = run(
        ["git", "config", "--get", "user.email"], cwd=root, check=False
    ).stdout.strip()
    if not name:
        run(["git", "config", "user.name", "autogen-bot"], cwd=root)
    if not email:
        run(
            ["git", "config", "user.email", "autogen@users.noreply.github.com"],
            cwd=root,
        )


def abort_in_progress_git_ops(root: Path) -> None:
    # Best-effort cleanup when a previous run died mid-git op.
    run(["git", "rebase", "--abort"], cwd=root, check=False)
    run(["git", "merge", "--abort"], cwd=root, check=False)
    run(["git", "cherry-pick", "--abort"], cwd=root, check=False)
    run(["git", "revert", "--abort"], cwd=root, check=False)


def hard_clean_repo(root: Path) -> None:
    abort_in_progress_git_ops(root)
    run(["git", "reset", "--hard", "HEAD"], cwd=root, check=False)
    run(["git", "clean", "-fd"], cwd=root, check=False)
    status = run(["git", "status", "--porcelain"], cwd=root).stdout.strip()
    if status:
        raise RuntimeError("Working tree not clean after reset/clean.")


def checkout_ref(root: Path, ref: str) -> None:
    abort_in_progress_git_ops(root)
    run(["git", "fetch", "origin"], cwd=root, check=False)
    run(["git", "checkout", "-f", ref], cwd=root, check=True)
    run(["git", "reset", "--hard", "HEAD"], cwd=root, check=False)
    run(["git", "clean", "-fd"], cwd=root, check=False)
    status = run(["git", "status", "--porcelain"], cwd=root).stdout.strip()
    if status:
        raise RuntimeError(f"Working tree not clean after checkout: {ref}")


def current_commit(root: Path) -> str:
    return run(["git", "rev-parse", "HEAD"], cwd=root).stdout.strip()


def state_schema_path(root: Path) -> Path:
    return root / "schemas" / "agent" / "scientist_state_event.schema.json"


def _event_common(*, campaign_id: str, seed_id: str, owner: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "campaign_id": campaign_id,
        "seed_id": seed_id,
        "owner": owner,
    }


def append_lease_event(
    *,
    repo_root: Path,
    state_repo_dir: Path,
    campaign_id: str,
    seed_id: str,
    stage: str,
    attempt: int,
    period_key: str,
    run_id: str,
    owner: str,
    base_commit: str,
) -> bool:
    identity = {
        "schema_version": 1,
        "event_type": "lease_acquired",
        "campaign_id": campaign_id,
        "seed_id": seed_id,
        "stage": stage,
        "attempt": int(attempt),
        "period_key": period_key,
    }
    event = {
        **_event_common(campaign_id=campaign_id, seed_id=seed_id, owner=owner),
        "event_id": event_id_for(identity),
        "event_type": "lease_acquired",
        "stage": stage,
        "attempt": int(attempt),
        "period_key": period_key,
        "run_id": run_id,
        "base_commit": base_commit,
        "started_at": utc_now_iso(),
    }
    try:
        append_event(
            repo_root=repo_root,
            state_repo_dir=state_repo_dir,
            campaign_id=campaign_id,
            event=event,
            schema_path=state_schema_path(repo_root),
            commit_message=f"state: lease {campaign_id} {stage} {seed_id} a{attempt}",
        )
    except LeaseAlreadyHeldError:
        return False
    return True


def append_heartbeat_event(
    *,
    repo_root: Path,
    state_repo_dir: Path,
    campaign_id: str,
    seed_id: str,
    stage: str,
    attempt: int,
    owner: str,
    heartbeat_bucket: int,
) -> None:
    identity = {
        "schema_version": 1,
        "event_type": "heartbeat",
        "campaign_id": campaign_id,
        "seed_id": seed_id,
        "stage": stage,
        "attempt": int(attempt),
        "heartbeat_bucket": int(heartbeat_bucket),
    }
    event = {
        **_event_common(campaign_id=campaign_id, seed_id=seed_id, owner=owner),
        "event_id": event_id_for(identity),
        "event_type": "heartbeat",
        "stage": stage,
        "attempt": int(attempt),
        "heartbeat_bucket": int(heartbeat_bucket),
        "last_heartbeat_at": utc_now_iso(),
    }
    append_event(
        repo_root=repo_root,
        state_repo_dir=state_repo_dir,
        campaign_id=campaign_id,
        event=event,
        schema_path=state_schema_path(repo_root),
        commit_message=f"state: heartbeat {campaign_id} {stage} {seed_id} a{attempt} b{heartbeat_bucket}",
    )


def append_finished_event(
    *,
    repo_root: Path,
    state_repo_dir: Path,
    campaign_id: str,
    seed_id: str,
    stage: str,
    attempt: int,
    period_key: str,
    run_id: str,
    owner: str,
    result: str,
    decision: str,
    metrics_summary: dict[str, Any],
    artifacts_ref: dict[str, Any],
    patch_ref: dict[str, str] | None = None,
    last_error: str = "",
) -> None:
    identity = {
        "schema_version": 1,
        "event_type": "finished",
        "campaign_id": campaign_id,
        "seed_id": seed_id,
        "stage": stage,
        "attempt": int(attempt),
        "run_id": run_id,
    }
    event: dict[str, Any] = {
        **_event_common(campaign_id=campaign_id, seed_id=seed_id, owner=owner),
        "event_id": event_id_for(identity),
        "event_type": "finished",
        "stage": stage,
        "attempt": int(attempt),
        "period_key": period_key,
        "run_id": run_id,
        "result": result,
        "decision": decision,
        "finished_at": utc_now_iso(),
        "metrics_summary": metrics_summary,
        "artifacts_ref": artifacts_ref,
    }
    if last_error:
        event["last_error"] = last_error
    if patch_ref:
        event["patch_ref"] = patch_ref

    append_event(
        repo_root=repo_root,
        state_repo_dir=state_repo_dir,
        campaign_id=campaign_id,
        event=event,
        schema_path=state_schema_path(repo_root),
        commit_message=f"state: finished {campaign_id} {stage} {seed_id} a{attempt} {result} {decision}",
    )


def append_reconciled_event(
    *,
    repo_root: Path,
    state_repo_dir: Path,
    campaign_id: str,
    seed_id: str,
    stage: str,
    attempt: int,
    owner: str,
    reconcile_action: str,
    reason: str,
) -> None:
    identity = {
        "schema_version": 1,
        "event_type": "reconciled",
        "campaign_id": campaign_id,
        "seed_id": seed_id,
        "stage": stage,
        "attempt": int(attempt),
        "reconcile_action": reconcile_action,
    }
    event = {
        **_event_common(campaign_id=campaign_id, seed_id=seed_id, owner=owner),
        "event_id": event_id_for(identity),
        "event_type": "reconciled",
        "stage": stage,
        "attempt": int(attempt),
        "reconcile_action": reconcile_action,
        "reconciled_at": utc_now_iso(),
        "reason": reason,
    }
    append_event(
        repo_root=repo_root,
        state_repo_dir=state_repo_dir,
        campaign_id=campaign_id,
        event=event,
        schema_path=state_schema_path(repo_root),
        commit_message=f"state: reconcile {campaign_id} {stage} {seed_id} a{attempt} {reconcile_action}",
    )


def reconcile_stale_leases(
    *,
    cfg: CampaignConfig,
    shard_i: int,
    max_tries: int,
    seeds: list[Seed],
    repo_root: Path,
    state_repo_dir: Path,
) -> int:
    """
    Reconcile stale leases for this shard by appending `reconciled` events.
    Returns number of reconciliations performed.
    """
    owner = canonical_owner()
    now = datetime.now(timezone.utc)
    events = load_events(state_repo_dir=state_repo_dir, campaign_id=cfg.campaign_id)
    idx = LedgerIndex(events)

    allowed = {
        s.seed_id
        for s in seeds
        if assigned_shard(s.seed_id, cfg.total_shards) == shard_i
    }
    reconciled_count = 0
    for lease in list(idx.leases.values()):
        if lease.seed_id not in allowed:
            continue
        key = (lease.seed_id, lease.stage, lease.attempt)
        if key in idx.finished or key in idx.reconciled:
            continue
        rules = stage_rules(cfg, lease.stage)
        last = idx.last_activity(lease)
        if now - last <= timedelta(seconds=int(rules.ttl_sec)):
            continue

        action = "requeue" if int(lease.attempt) < int(max_tries) else "fail_permanent"
        append_reconciled_event(
            repo_root=repo_root,
            state_repo_dir=state_repo_dir,
            campaign_id=cfg.campaign_id,
            seed_id=lease.seed_id,
            stage=lease.stage,
            attempt=int(lease.attempt),
            owner=owner,
            reconcile_action=action,
            reason=(
                f"stale lease: last_activity={last.strftime('%Y-%m-%dT%H:%M:%SZ')} "
                f"ttl_sec={int(rules.ttl_sec)}"
            ),
        )
        reconciled_count += 1
    return reconciled_count


def run_experiment_with_heartbeat(
    *,
    repo_root: Path,
    state_repo_dir: Path,
    campaign_id: str,
    seed_id: str,
    stage: str,
    attempt: int,
    run_id: str,
    period_key: str,
    out_root: Path,
    plan_path: Path,
    bootstrap_b: int,
    prompt: str,
    profile: str,
    apply_patch: Path | None,
    heartbeat_bucket_sec: int,
) -> None:
    owner = canonical_owner()
    cmd = [
        sys.executable,
        "scripts/agent/run_experiment.py",
        "--plan",
        str(plan_path),
        "--profile",
        profile,
        "--prompt",
        prompt,
        "--results-only",
        "--out-dir",
        str(out_root),
        "--bootstrap-b",
        str(int(bootstrap_b)),
    ]
    if apply_patch is not None:
        cmd.extend(["--apply-patch", str(apply_patch)])

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    proc = subprocess.Popen(cmd, cwd=repo_root, env=env)
    last_bucket: int | None = None
    try:
        while proc.poll() is None:
            bucket = int(time.time() // max(int(heartbeat_bucket_sec), 60))
            if last_bucket is None or bucket != last_bucket:
                # Best-effort: if heartbeat push fails, keep going.
                try:
                    append_heartbeat_event(
                        repo_root=repo_root,
                        state_repo_dir=state_repo_dir,
                        campaign_id=campaign_id,
                        seed_id=seed_id,
                        stage=stage,
                        attempt=int(attempt),
                        owner=owner,
                        heartbeat_bucket=bucket,
                    )
                except Exception:
                    pass
                last_bucket = bucket
            time.sleep(60)
    finally:
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"run_experiment failed ({rc}) for {run_id}")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stage_outputs_dir(out_root: Path, run_id: str) -> Path:
    return out_root / "experiments" / run_id


def patch_bundle_paths(exp_dir: Path) -> tuple[Path, str] | None:
    diff_path = exp_dir / "patch.diff"
    sha_path = exp_dir / "patch.sha256"
    if not (diff_path.exists() and sha_path.exists()):
        return None
    sha = sha_path.read_text(encoding="utf-8").strip()
    if not sha:
        return None
    return diff_path, sha


def maybe_store_patch_to_ledger(
    *,
    repo_root: Path,
    state_repo_dir: Path,
    campaign_id: str,
    seed_id: str,
    patch_sha256: str,
    patch_path: Path,
) -> dict[str, str]:
    stored = store_patch(
        repo_root=repo_root,
        state_repo_dir=state_repo_dir,
        campaign_id=campaign_id,
        seed_id=seed_id,
        patch_sha256=patch_sha256,
        patch_src=patch_path,
        commit_message=f"state: patch {campaign_id} {seed_id} {patch_sha256}",
    )
    # Store paths relative to the state worktree root for portability.
    rel = str(Path(stored).relative_to(state_repo_dir)).replace("\\", "/")
    return {"sha256": patch_sha256, "path": rel}


def current_branch(root: Path) -> str:
    return run(
        ["git", "branch", "--show-current"], cwd=root, check=False
    ).stdout.strip()


def delete_branch(root: Path, branch: str) -> None:
    if not branch:
        return
    run(["git", "branch", "-D", branch], cwd=root, check=False)


def parse_date_ymd(s: str):
    return datetime.strptime(str(s).strip(), "%Y-%m-%d").date()


def run_stage_job(
    *,
    cfg: CampaignConfig,
    seed: Seed,
    stage: str,
    attempt: int,
    shard_i: int,
    repo_root: Path,
    state_repo_dir: Path,
    out_root: Path,
    base_ref: str,
    prompt: str,
    profile: str,
    max_tries: int,
    upstream_run_id: str = "",
    upstream_patch_ref: dict[str, str] | None = None,
    upstream_base_commit: str = "",
) -> bool:
    """
    Execute one seed at a given stage, write results-only artifacts, and append ledger events.
    """
    if assigned_shard(seed.seed_id, cfg.total_shards) != shard_i:
        raise RuntimeError(f"Seed {seed.seed_id} is not assigned to shard {shard_i}.")
    if attempt < 1 or attempt > max_tries:
        raise RuntimeError(f"Invalid attempt {attempt} (max_tries={max_tries}).")

    rules = stage_rules(cfg, stage)
    period_key, test_start, test_end = cfg.stage_period(stage)
    splits = cfg.build_splits_for_test_start(parse_date_ymd(test_start))
    owner = canonical_owner()

    run_id = _safe_run_id(
        campaign_id=cfg.campaign_id,
        stage=stage,
        seed_id=seed.seed_id,
        attempt=int(attempt),
    )
    eval_cmd = build_holdout_command(
        run_id=run_id, splits=splits, test_start=test_start, test_end=test_end
    )
    plan = build_plan(
        seed=seed,
        run_id=run_id,
        eval_command=eval_cmd,
        reason=f"scientist_v2 {cfg.campaign_id} {stage} shard={shard_i}/{cfg.total_shards}",
    )

    plans_dir = ensure_dir(out_root / "plans" / stage)
    plan_path = plans_dir / f"{run_id}.json"
    plan_path.write_text(
        json.dumps(plan, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )

    branch_before = current_branch(repo_root)
    branch_after = ""
    base_commit = ""
    try:
        ensure_git_identity(repo_root)

        if stage == "stage1":
            checkout_ref(repo_root, base_ref)
        else:
            if not upstream_patch_ref or not upstream_patch_ref.get("path"):
                raise RuntimeError(
                    f"Missing upstream_patch_ref for {stage} ({seed.seed_id})."
                )
            if not upstream_base_commit:
                raise RuntimeError(
                    f"Missing upstream_base_commit for {stage} ({seed.seed_id})."
                )
            checkout_ref(repo_root, upstream_base_commit)

        base_commit = current_commit(repo_root)

        lease_ok = append_lease_event(
            repo_root=repo_root,
            state_repo_dir=state_repo_dir,
            campaign_id=cfg.campaign_id,
            seed_id=seed.seed_id,
            stage=stage,
            attempt=int(attempt),
            period_key=period_key,
            run_id=run_id,
            owner=owner,
            base_commit=base_commit,
        )
        if not lease_ok:
            print(
                f"[skip] lease already held: campaign={cfg.campaign_id} stage={stage} "
                f"seed={seed.seed_id} attempt={int(attempt)}"
            )
            return False

        patch_path: Path | None = None
        patch_ref: dict[str, str] | None = None
        if stage != "stage1":
            patch_path = state_repo_dir / Path(str(upstream_patch_ref["path"]))
            if not patch_path.exists():
                raise RuntimeError(f"Upstream patch not found: {patch_path}")
            patch_ref = dict(upstream_patch_ref)

        run_experiment_with_heartbeat(
            repo_root=repo_root,
            state_repo_dir=state_repo_dir,
            campaign_id=cfg.campaign_id,
            seed_id=seed.seed_id,
            stage=stage,
            attempt=int(attempt),
            run_id=run_id,
            period_key=period_key,
            out_root=out_root,
            plan_path=plan_path,
            bootstrap_b=int(rules.bootstrap_B),
            prompt=prompt,
            profile=profile,
            apply_patch=patch_path,
            heartbeat_bucket_sec=int(cfg.central_heartbeat_bucket_sec),
        )

        exp_dir = stage_outputs_dir(out_root, run_id)
        result_path = exp_dir / "experiment_result.json"
        if not result_path.exists():
            raise RuntimeError(f"experiment_result.json not found: {result_path}")
        result = read_json(result_path)

        if stage == "stage1":
            bundle = patch_bundle_paths(exp_dir)
            if not bundle:
                raise RuntimeError(
                    f"patch bundle not found in experiment dir: {exp_dir}"
                )
            diff_path, patch_sha = bundle
            patch_ref = maybe_store_patch_to_ledger(
                repo_root=repo_root,
                state_repo_dir=state_repo_dir,
                campaign_id=cfg.campaign_id,
                seed_id=seed.seed_id,
                patch_sha256=patch_sha,
                patch_path=diff_path,
            )

        metrics_summary = metrics_summary_from_result(result)
        decision = evaluate_stage_decision(stage=stage, rules=rules, result=result)

        artifacts_ref: dict[str, Any] = {
            "out_root": str(out_root).replace("\\", "/"),
            "experiment_dir": str(exp_dir).replace("\\", "/"),
            "experiment_result": str(result_path).replace("\\", "/"),
            "upstream_run_id": upstream_run_id,
            "upstream_base_commit": upstream_base_commit,
            "shard": f"{shard_i}/{cfg.total_shards}",
            "github": {
                "run_id": (os.environ.get("GITHUB_RUN_ID") or "").strip(),
                "run_attempt": (os.environ.get("GITHUB_RUN_ATTEMPT") or "").strip(),
                "job": (os.environ.get("GITHUB_JOB") or "").strip(),
            },
        }

        append_finished_event(
            repo_root=repo_root,
            state_repo_dir=state_repo_dir,
            campaign_id=cfg.campaign_id,
            seed_id=seed.seed_id,
            stage=stage,
            attempt=int(attempt),
            period_key=period_key,
            run_id=run_id,
            owner=owner,
            result="success",
            decision=decision,
            metrics_summary=metrics_summary,
            artifacts_ref=artifacts_ref,
            patch_ref=patch_ref,
        )
        return True

    except Exception as exc:
        # Best-effort: record failure for this attempt.
        try:
            artifacts_ref = {
                "out_root": str(out_root).replace("\\", "/"),
                "shard": f"{shard_i}/{cfg.total_shards}",
                "upstream_run_id": upstream_run_id,
                "upstream_base_commit": upstream_base_commit,
            }
            append_finished_event(
                repo_root=repo_root,
                state_repo_dir=state_repo_dir,
                campaign_id=cfg.campaign_id,
                seed_id=seed.seed_id,
                stage=stage,
                attempt=int(attempt),
                period_key=period_key,
                run_id=run_id,
                owner=owner,
                result="failure",
                decision="iterate",
                metrics_summary=_null_metrics_summary(),
                artifacts_ref=artifacts_ref,
                last_error=str(exc),
                patch_ref=upstream_patch_ref if stage != "stage1" else None,
            )
        except Exception:
            pass
        raise
    finally:
        # Cleanup: return to a predictable ref and delete the agent branch created by run_experiment.
        try:
            branch_after = current_branch(repo_root)
            checkout_ref(repo_root, base_ref)
            if branch_after and branch_after.startswith("agent/"):
                delete_branch(repo_root, branch_after)
            # Restore original branch if we started from one (local runs).
            if branch_before and branch_before != current_branch(repo_root):
                checkout_ref(repo_root, branch_before)
        except Exception:
            # Avoid masking the original exception.
            pass


def select_next_stage1(
    *,
    seeds: list[Seed],
    idx: LedgerIndex,
    cfg: CampaignConfig,
    shard_i: int,
    max_tries: int,
) -> tuple[Seed, int] | None:
    for seed in seeds:
        if assigned_shard(seed.seed_id, cfg.total_shards) != shard_i:
            continue
        if idx.has_success(seed.seed_id, "stage1"):
            continue
        if idx.active_lease(seed.seed_id, "stage1") is not None:
            continue
        attempt = idx.next_attempt(seed.seed_id, "stage1")
        if attempt > int(max_tries):
            continue
        return seed, attempt
    return None


def select_next_stage2(
    *,
    seeds: list[Seed],
    idx: LedgerIndex,
    cfg: CampaignConfig,
    shard_i: int,
    max_tries: int,
) -> tuple[Seed, int, Finished] | None:
    for seed in seeds:
        if assigned_shard(seed.seed_id, cfg.total_shards) != shard_i:
            continue
        if idx.has_success(seed.seed_id, "stage2"):
            continue
        if idx.active_lease(seed.seed_id, "stage2") is not None:
            continue
        src = idx.latest_success(seed.seed_id, "stage1", decision="candidate")
        if src is None:
            continue
        attempt = idx.next_attempt(seed.seed_id, "stage2")
        if attempt > int(max_tries):
            continue
        return seed, attempt, src
    return None


def select_next_holdout(
    *,
    seeds: list[Seed],
    idx: LedgerIndex,
    cfg: CampaignConfig,
    shard_i: int,
    max_tries: int,
) -> tuple[Seed, int, Finished] | None:
    for seed in seeds:
        if assigned_shard(seed.seed_id, cfg.total_shards) != shard_i:
            continue
        if idx.has_success(seed.seed_id, "holdout"):
            continue
        if idx.active_lease(seed.seed_id, "holdout") is not None:
            continue
        src = idx.latest_success(seed.seed_id, "stage2", decision="accept")
        if src is None:
            continue
        attempt = idx.next_attempt(seed.seed_id, "holdout")
        if attempt > int(max_tries):
            continue
        return seed, attempt, src
    return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--campaign",
        required=True,
        help="Campaign ID (config/scientist_campaigns/<id>.yml)",
    )
    p.add_argument("--stage", required=True, choices=["stage1", "stage2", "holdout"])
    p.add_argument("--shard", default="0/1", help="Shard in i/n form (e.g. 0/2)")
    p.add_argument(
        "--max-jobs",
        type=int,
        default=5,
        help="Max seeds to process in this invocation",
    )
    p.add_argument("--max-tries", type=int, default=3)
    p.add_argument("--seed-path", default="experiments/seed_hypotheses.yaml")
    p.add_argument(
        "--base-ref",
        default="origin/main",
        help="Git ref to reset to between runs (stage1 base)",
    )
    p.add_argument("--prompt", default="prompts/agent/fixer_implement.md")
    p.add_argument("--profile", default="agent_loop")
    p.add_argument(
        "--artifact-root",
        default="",
        help="Base artifact dir (defaults to $KEIBA_ARTIFACT_DIR)",
    )
    p.add_argument(
        "--state-repo-dir",
        default="",
        help="State worktree dir (defaults to $KEIBA_STATE_REPO_DIR)",
    )
    p.add_argument("--only-seed", default="", help="If set, only process this seed_id")
    args = p.parse_args()

    root = repo_root()
    cfg = load_campaign_by_id(root, str(args.campaign).strip())

    shard_i, shard_n = parse_shard(args.shard)
    if shard_n != int(cfg.total_shards):
        raise RuntimeError(
            f"Shard mismatch: got {shard_i}/{shard_n} but campaign requires total_shards={cfg.total_shards}"
        )

    seed_path = root / str(args.seed_path)
    seeds = load_seeds(seed_path)
    if args.only_seed:
        seeds = [s for s in seeds if s.seed_id == str(args.only_seed).strip()]
        if not seeds:
            raise RuntimeError(f"--only-seed not found in seed list: {args.only_seed}")

    state_repo_dir_raw = (
        str(args.state_repo_dir) or os.environ.get("KEIBA_STATE_REPO_DIR") or ""
    ).strip()
    if not state_repo_dir_raw:
        raise RuntimeError("Set --state-repo-dir or $KEIBA_STATE_REPO_DIR")
    state_repo_dir = Path(state_repo_dir_raw)
    ensure_state_worktree(root=root, worktree_dir=state_repo_dir)

    artifact_root_raw = (
        str(args.artifact_root) or os.environ.get("KEIBA_ARTIFACT_DIR") or ""
    ).strip()
    if not artifact_root_raw:
        raise RuntimeError("Set --artifact-root or $KEIBA_ARTIFACT_DIR")
    artifact_root = Path(artifact_root_raw)
    if not artifact_root.is_absolute():
        artifact_root = (root / artifact_root).resolve()
    out_root = ensure_dir(artifact_root / cfg.campaign_id)

    ensure_git_identity(root)
    hard_clean_repo(root)

    # Reconcile stale leases once at startup.
    reconcile_stale_leases(
        cfg=cfg,
        shard_i=shard_i,
        max_tries=int(args.max_tries),
        seeds=seeds,
        repo_root=root,
        state_repo_dir=state_repo_dir,
    )

    processed = 0
    max_jobs = max(int(args.max_jobs), 0)
    spins = 0
    max_spins = max_jobs * 10 if max_jobs else 0
    while processed < max_jobs:
        spins += 1
        if max_spins and spins > max_spins:
            print(
                f"[warn] too many iterations without progress; stopping "
                f"(processed={processed} spins={spins} max_jobs={max_jobs})"
            )
            break
        events = load_events(state_repo_dir=state_repo_dir, campaign_id=cfg.campaign_id)
        idx = LedgerIndex(events)

        if args.stage == "stage1":
            pick = select_next_stage1(
                seeds=seeds,
                idx=idx,
                cfg=cfg,
                shard_i=shard_i,
                max_tries=int(args.max_tries),
            )
            if not pick:
                break
            seed, attempt = pick
            ran = run_stage_job(
                cfg=cfg,
                seed=seed,
                stage="stage1",
                attempt=int(attempt),
                shard_i=shard_i,
                repo_root=root,
                state_repo_dir=state_repo_dir,
                out_root=out_root,
                base_ref=str(args.base_ref),
                prompt=str(args.prompt),
                profile=str(args.profile),
                max_tries=int(args.max_tries),
            )
            if ran:
                processed += 1
            else:
                time.sleep(1)
            continue

        if args.stage == "stage2":
            pick2 = select_next_stage2(
                seeds=seeds,
                idx=idx,
                cfg=cfg,
                shard_i=shard_i,
                max_tries=int(args.max_tries),
            )
            if not pick2:
                break
            seed, attempt, src = pick2
            lease = idx.lease_for_run(src.run_id)
            if lease is None:
                raise RuntimeError(
                    f"stage1 lease not found for upstream run: {src.run_id}"
                )
            ran = run_stage_job(
                cfg=cfg,
                seed=seed,
                stage="stage2",
                attempt=int(attempt),
                shard_i=shard_i,
                repo_root=root,
                state_repo_dir=state_repo_dir,
                out_root=out_root,
                base_ref=str(args.base_ref),
                prompt=str(args.prompt),
                profile=str(args.profile),
                max_tries=int(args.max_tries),
                upstream_run_id=src.run_id,
                upstream_patch_ref=src.patch_ref,
                upstream_base_commit=lease.base_commit,
            )
            if ran:
                processed += 1
            else:
                time.sleep(1)
            continue

        if args.stage == "holdout":
            pick3 = select_next_holdout(
                seeds=seeds,
                idx=idx,
                cfg=cfg,
                shard_i=shard_i,
                max_tries=int(args.max_tries),
            )
            if not pick3:
                break
            seed, attempt, src = pick3
            lease = idx.lease_for_run(src.run_id)
            if lease is None:
                raise RuntimeError(
                    f"stage2 lease not found for upstream run: {src.run_id}"
                )
            ran = run_stage_job(
                cfg=cfg,
                seed=seed,
                stage="holdout",
                attempt=int(attempt),
                shard_i=shard_i,
                repo_root=root,
                state_repo_dir=state_repo_dir,
                out_root=out_root,
                base_ref=str(args.base_ref),
                prompt=str(args.prompt),
                profile=str(args.profile),
                max_tries=int(args.max_tries),
                upstream_run_id=src.run_id,
                upstream_patch_ref=src.patch_ref,
                upstream_base_commit=lease.base_commit,
            )
            if ran:
                processed += 1
            else:
                time.sleep(1)
            continue

        raise RuntimeError(f"Unhandled stage: {args.stage}")

    print(
        json.dumps(
            {
                "campaign_id": cfg.campaign_id,
                "stage": str(args.stage),
                "shard": f"{shard_i}/{shard_n}",
                "processed": processed,
                "out_root": str(out_root).replace("\\", "/"),
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
