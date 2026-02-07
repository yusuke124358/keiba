#!/usr/bin/env python3
"""
Scientist Loop V2 - Promotion

Reads the central append-only ledger (scientist-state) and creates PRs only for
holdout-accepted seeds. This is intentionally separate from exploration runs to
avoid PR flood.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except Exception as exc:  # pragma: no cover
    raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from exc

from scientist_campaign import CampaignConfig, load_campaign_by_id
from state_ledger import (
    append_event,
    compute_event_id,
    load_events,
    owner_string,
    utc_now_iso,
)
from state_worktree import ensure_git_identity, ensure_state_worktree


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


def state_schema_path(root: Path) -> Path:
    return root / "schemas" / "agent" / "scientist_state_event.schema.json"


def _slug(text: str) -> str:
    s = str(text or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return s.strip("-") or "promotion"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_seeds(path: Path) -> dict[str, dict[str, Any]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise RuntimeError(f"seed_hypotheses.yaml must be a list: {path}")
    out: dict[str, dict[str, Any]] = {}
    for item in raw:
        if not isinstance(item, dict):
            continue
        sid = str(item.get("id") or "").strip()
        if sid:
            out[sid] = item
    return out


def _build_holdout_eval_command(*, cfg: CampaignConfig, run_id: str) -> str:
    _, test_start, test_end = cfg.stage_period("holdout")
    splits = cfg.build_splits_for_test_start(_parse_date_ymd(test_start))
    return (
        "py64_analysis/scripts/run_holdout.py"
        f" --train-start {splits['train_start']} --train-end {splits['train_end']}"
        f" --valid-start {splits['valid_start']} --valid-end {splits['valid_end']}"
        f" --test-start {test_start} --test-end {test_end}"
        f" --name {run_id} --out-dir data/holdout_runs/{run_id}"
    )


def _build_plan(
    *, seed: dict[str, Any], run_id: str, eval_command: str
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "seed_id": str(seed.get("id") or "").strip(),
        "title": str(seed.get("title") or "").strip(),
        "hypothesis": str(seed.get("hypothesis") or "").strip(),
        "change_scope": str(seed.get("change_scope") or "").strip(),
        "acceptance_criteria": [
            str(x) for x in (seed.get("acceptance_criteria") or [])
        ],
        "metrics": [str(x) for x in (seed.get("metrics") or [])],
        "risk_level": str(seed.get("risk_level") or "medium").strip(),
        "max_diff_size": int(seed.get("max_diff_size") or 1),
        "eval_command": [eval_command],
        "metrics_path": f"data/holdout_runs/{run_id}/metrics.json",
        "decision": "do",
        "reason": "promotion recheck on latest main (holdout window)",
    }


def _parse_date_ymd(s: str):
    from datetime import datetime

    return datetime.strptime(str(s).strip(), "%Y-%m-%d").date()


@dataclass(frozen=True)
class HoldoutAccept:
    seed_id: str
    run_id: str
    patch_path: str
    patch_sha256: str
    finished_at: str


def _collect_holdout_accepts(events: list[dict[str, Any]]) -> list[HoldoutAccept]:
    out: list[HoldoutAccept] = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        if str(ev.get("event_type") or "").strip() != "finished":
            continue
        if str(ev.get("stage") or "").strip() != "holdout":
            continue
        if str(ev.get("result") or "").strip() != "success":
            continue
        if str(ev.get("decision") or "").strip() != "accept":
            continue
        patch_ref = (
            ev.get("patch_ref") if isinstance(ev.get("patch_ref"), dict) else None
        )
        if not patch_ref:
            continue
        seed_id = str(ev.get("seed_id") or "").strip()
        run_id = str(ev.get("run_id") or "").strip()
        path = str(patch_ref.get("path") or "").strip()
        sha = str(patch_ref.get("sha256") or "").strip()
        finished_at = str(ev.get("finished_at") or "").strip()
        if not (seed_id and run_id and path and sha):
            continue
        out.append(
            HoldoutAccept(
                seed_id=seed_id,
                run_id=run_id,
                patch_path=path,
                patch_sha256=sha,
                finished_at=finished_at,
            )
        )
    # Newest first, to promote recent acceptances first.
    return sorted(out, key=lambda x: x.finished_at, reverse=True)


def _promoted_run_ids(events: list[dict[str, Any]]) -> set[str]:
    out: set[str] = set()
    for ev in events:
        if not isinstance(ev, dict):
            continue
        if str(ev.get("event_type") or "").strip() != "promotion_applied":
            continue
        rid = str(ev.get("run_id") or "").strip()
        if rid:
            out.add(rid)
    return out


def _append_promotion_event(
    *,
    root: Path,
    state_repo_dir: Path,
    campaign_id: str,
    seed_id: str,
    event_type: str,
    run_id: str,
    reason: str,
) -> None:
    identity = {
        "schema_version": 1,
        "event_type": event_type,
        "campaign_id": campaign_id,
        "seed_id": seed_id,
        "run_id": run_id,
    }
    event: dict[str, Any] = {
        "schema_version": 1,
        "event_id": compute_event_id(identity),
        "event_type": event_type,
        "campaign_id": campaign_id,
        "seed_id": seed_id,
        "owner": owner_string(),
        "promotion_at": utc_now_iso(),
        "run_id": run_id,
        "reason": reason,
    }
    append_event(
        repo_root=root,
        state_repo_dir=state_repo_dir,
        campaign_id=campaign_id,
        event=event,
        schema_path=state_schema_path(root),
        commit_message=f"state: {event_type} {campaign_id} {seed_id} {run_id}",
    )


def _hard_clean_repo(root: Path) -> None:
    run(["git", "rebase", "--abort"], cwd=root, check=False, capture_output=True)
    run(["git", "merge", "--abort"], cwd=root, check=False, capture_output=True)
    run(["git", "cherry-pick", "--abort"], cwd=root, check=False, capture_output=True)
    run(["git", "reset", "--hard", "HEAD"], cwd=root, check=False, capture_output=True)
    run(["git", "clean", "-fd"], cwd=root, check=False, capture_output=True)
    status = run(["git", "status", "--porcelain"], cwd=root).stdout.strip()
    if status:
        raise RuntimeError("Working tree not clean after reset/clean.")


def _checkout_base(root: Path, base_ref: str) -> None:
    # Ensure we recheck/publish against the latest remote tip (self-hosted runners can have stale local branches).
    base_ref = str(base_ref or "").strip()
    if not base_ref:
        raise RuntimeError("base_ref is required.")
    run(["git", "fetch", "origin"], cwd=root, check=False, capture_output=True)
    remote = base_ref if base_ref.startswith("origin/") else f"origin/{base_ref}"
    local = base_ref[len("origin/") :] if base_ref.startswith("origin/") else base_ref
    # Make the local base branch match the remote tip deterministically.
    run(
        ["git", "checkout", "-B", local, remote],
        cwd=root,
        check=True,
        capture_output=True,
    )
    _hard_clean_repo(root)


def _current_branch(root: Path) -> str:
    return run(["git", "branch", "--show-current"], cwd=root).stdout.strip()


def _branch_exists_remote(root: Path, branch: str) -> bool:
    result = run(
        ["git", "ls-remote", "--heads", "origin", branch],
        cwd=root,
        check=False,
        capture_output=True,
    )
    return bool((result.stdout or "").strip())


def _pr_exists_for_branch(root: Path, branch: str, env: dict[str, str]) -> bool:
    code = subprocess.run(
        ["gh", "pr", "view", "--head", branch, "--json", "number", "-q", ".number"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return code.returncode == 0 and (code.stdout or "").strip().isdigit()


def _get_pr_number(root: Path, branch: str, env: dict[str, str]) -> str:
    out = run(
        ["gh", "pr", "view", "--head", branch, "--json", "number", "-q", ".number"],
        cwd=root,
        env=env,
        capture_output=True,
    ).stdout.strip()
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--campaign",
        required=True,
        help="Campaign ID (config/scientist_campaigns/<id>.yml)",
    )
    p.add_argument("--base", default="main", help="Base branch for promotion PRs")
    p.add_argument("--max-prs", type=int, default=3, help="Max PRs to create per run")
    p.add_argument("--seed-path", default="experiments/seed_hypotheses.yaml")
    p.add_argument(
        "--state-repo-dir",
        default="",
        help="State worktree dir (defaults to $KEIBA_STATE_REPO_DIR)",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    root = repo_root()
    cfg = load_campaign_by_id(root, str(args.campaign).strip())

    state_repo_dir_raw = (
        str(args.state_repo_dir) or os.environ.get("KEIBA_STATE_REPO_DIR") or ""
    ).strip()
    if not state_repo_dir_raw:
        raise RuntimeError("Set --state-repo-dir or $KEIBA_STATE_REPO_DIR")
    state_repo_dir = Path(state_repo_dir_raw)
    ensure_state_worktree(root=root, worktree_dir=state_repo_dir)

    seed_map = _load_seeds(root / str(args.seed_path))

    events = load_events(state_repo_dir=state_repo_dir, campaign_id=cfg.campaign_id)
    accepts = _collect_holdout_accepts(events)
    promoted = _promoted_run_ids(events)

    env = os.environ.copy()
    token = env.get("AUTO_FIX_PUSH_TOKEN") or env.get("GH_TOKEN")
    if token:
        env["GH_TOKEN"] = token

    created = 0
    for acc in accepts:
        if created >= max(int(args.max_prs), 0):
            break
        if acc.run_id in promoted:
            continue

        seed = seed_map.get(acc.seed_id)
        if not seed:
            _append_promotion_event(
                root=root,
                state_repo_dir=state_repo_dir,
                campaign_id=cfg.campaign_id,
                seed_id=acc.seed_id,
                event_type="promotion_skipped_recheck_fail",
                run_id=acc.run_id,
                reason="seed not found in seed_hypotheses.yaml",
            )
            continue

        patch_path = state_repo_dir / Path(acc.patch_path)
        if not patch_path.exists():
            _append_promotion_event(
                root=root,
                state_repo_dir=state_repo_dir,
                campaign_id=cfg.campaign_id,
                seed_id=acc.seed_id,
                event_type="promotion_skipped_apply_fail",
                run_id=acc.run_id,
                reason=f"patch not found: {patch_path}",
            )
            continue

        actual_sha256 = _sha256_file(patch_path)
        if actual_sha256 != acc.patch_sha256:
            _append_promotion_event(
                root=root,
                state_repo_dir=state_repo_dir,
                campaign_id=cfg.campaign_id,
                seed_id=acc.seed_id,
                event_type="promotion_skipped_hash_mismatch",
                run_id=acc.run_id,
                reason=(
                    "patch sha256 mismatch: "
                    f"expected={acc.patch_sha256} actual={actual_sha256} path={patch_path}"
                ),
            )
            continue

        branch = f"promotion/{cfg.campaign_id}/{acc.seed_id}/{acc.run_id}"
        branch = branch.replace("\\", "/")

        # Avoid duplicates if branch/PR already exists.
        if _branch_exists_remote(root, branch):
            if _pr_exists_for_branch(root, branch, env):
                _append_promotion_event(
                    root=root,
                    state_repo_dir=state_repo_dir,
                    campaign_id=cfg.campaign_id,
                    seed_id=acc.seed_id,
                    event_type="promotion_applied",
                    run_id=acc.run_id,
                    reason=f"PR already exists for branch={branch}",
                )
                created += 1
                continue
            _append_promotion_event(
                root=root,
                state_repo_dir=state_repo_dir,
                campaign_id=cfg.campaign_id,
                seed_id=acc.seed_id,
                event_type="promotion_skipped_conflict",
                run_id=acc.run_id,
                reason=f"remote branch already exists without PR: {branch}",
            )
            continue

        title = str(seed.get("title") or "").strip()
        pr_title = f"promotion: {cfg.campaign_id} {acc.seed_id} {title}".strip()

        if args.dry_run:
            print(
                json.dumps(
                    {
                        "action": "would_promote",
                        "campaign_id": cfg.campaign_id,
                        "seed_id": acc.seed_id,
                        "run_id": acc.run_id,
                        "branch": branch,
                        "patch": str(patch_path).replace("\\", "/"),
                    },
                    ensure_ascii=True,
                )
            )
            continue

        # Prepare branch on latest base.
        _checkout_base(root, args.base)
        ensure_git_identity(root)
        run(["git", "checkout", "-b", branch], cwd=root, capture_output=True)
        _hard_clean_repo(root)

        # Re-check evaluation on latest base with the stored patch.
        plan_dir = root / "artifacts" / "agent" / "promotion_plans"
        plan_dir.mkdir(parents=True, exist_ok=True)
        eval_cmd = _build_holdout_eval_command(cfg=cfg, run_id=acc.run_id)
        plan = _build_plan(seed=seed, run_id=acc.run_id, eval_command=eval_cmd)
        plan_path = plan_dir / f"{acc.run_id}.json"
        plan_path.write_text(
            json.dumps(plan, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
        )

        # Use current interpreter (venv in workflow) and do not create an agent/* branch.
        bootstrap_b = int(cfg.holdout.bootstrap_B)
        run(
            [
                sys.executable,
                "scripts/agent/run_experiment.py",
                "--plan",
                str(plan_path),
                "--no-checkout-branch",
                "--apply-patch",
                str(patch_path),
                "--bootstrap-b",
                str(bootstrap_b),
            ],
            cwd=root,
            env={**env, "PYTHONIOENCODING": "utf-8"},
            capture_output=False,
        )

        result_json = root / "experiments" / "runs" / f"{acc.run_id}.json"
        if not result_json.exists():
            _append_promotion_event(
                root=root,
                state_repo_dir=state_repo_dir,
                campaign_id=cfg.campaign_id,
                seed_id=acc.seed_id,
                event_type="promotion_skipped_recheck_fail",
                run_id=acc.run_id,
                reason=f"missing result json after recheck: {result_json}",
            )
            continue

        result = json.loads(result_json.read_text(encoding="utf-8"))
        decision = (
            ((result.get("decision") or {}).get("decision") or "").strip().lower()
        )
        if decision != "accept":
            _append_promotion_event(
                root=root,
                state_repo_dir=state_repo_dir,
                campaign_id=cfg.campaign_id,
                seed_id=acc.seed_id,
                event_type="promotion_skipped_recheck_fail",
                run_id=acc.run_id,
                reason=f"recheck decision={decision}",
            )
            # Do not create a PR; leave branch local only.
            _checkout_base(root, args.base)
            run(["git", "branch", "-D", branch], cwd=root, check=False)
            continue

        ms = result.get("stats") or {}
        ci = ms.get("delta_roi_ci95") or [None, None]
        p_one_sided = ms.get("p_one_sided_delta_le_0")
        metrics = result.get("metrics") or {}
        b_metrics = result.get("baseline_metrics") or {}
        delta = (result.get("deltas") or {}).get("delta_roi")

        def _profit(stake: Any, total_return: Any) -> float | None:
            if stake is None or total_return is None:
                return None
            try:
                return float(total_return) - float(stake)
            except Exception:
                return None

        def _split_period(x: Any) -> tuple[str, str]:
            s = str(x or "").strip()
            if " to " in s:
                a, b = s.split(" to ", 1)
                return a.strip(), b.strip()
            return "N/A", "N/A"

        v_stake = metrics.get("total_stake")
        v_return = metrics.get("total_return")
        v_profit = _profit(v_stake, v_return)
        v_dd = metrics.get("max_drawdown")
        v_start, v_end = _split_period(metrics.get("test_period"))

        b_stake = b_metrics.get("total_stake")
        b_return = b_metrics.get("total_return")
        b_profit = _profit(b_stake, b_return)
        b_dd = b_metrics.get("max_drawdown")
        b_start, b_end = _split_period(b_metrics.get("test_period"))

        pr_body = "\n".join(
            [
                f"Campaign: `{cfg.campaign_id}`",
                f"Seed: `{acc.seed_id}`",
                f"Holdout run_id: `{acc.run_id}`",
                "Decision (recheck): `accept`",
                "",
                "## Summary (holdout recheck)",
                f"- ROI (variant): {metrics.get('roi')}",
                f"- ROI (baseline): {b_metrics.get('roi')}",
                f"- Stake (variant): {v_stake}",
                f"- Return (variant): {v_return}",
                f"- Profit (variant): {v_profit}",
                f"- Max drawdown (variant): {v_dd}",
                f"- Stake (baseline): {b_stake}",
                f"- Return (baseline): {b_return}",
                f"- Profit (baseline): {b_profit}",
                f"- Max drawdown (baseline): {b_dd}",
                f"- delta_ROI: {delta}",
                f"- delta_ROI_CI95: {ci}",
                f"- p_one_sided_delta_le_0: {p_one_sided}",
                f"- n_bets: {metrics.get('n_bets')}",
                f"- n_days: {metrics.get('n_days')}",
                f"- Test start (variant): {v_start}",
                f"- Test end (variant): {v_end}",
                f"- Test start (baseline): {b_start}",
                f"- Test end (baseline): {b_end}",
                "- ROI definition: ROI = profit / stake, profit = return - stake.",
                "",
                "## Details",
                f"- Experiment log: `docs/experiments/{acc.run_id}.md`",
                f"- Result JSON: `experiments/runs/{acc.run_id}.json`",
                "",
                "This PR was auto-created by Scientist Loop V2 promotion. It rechecks holdout on the current base branch and only opens a PR if it still passes.",
            ]
        )

        body_file = (
            root / "artifacts" / "agent" / f"promotion_pr_body_{_slug(acc.run_id)}.md"
        )
        body_file.parent.mkdir(parents=True, exist_ok=True)
        body_file.write_text(pr_body + "\n", encoding="utf-8")

        # Publish PR.
        labels = ",".join(cfg.promotion_labels)
        run(
            [
                sys.executable,
                "scripts/publish/publisher.py",
                "--base",
                str(args.base),
                "--title",
                pr_title,
                "--body-file",
                str(body_file),
                "--labels",
                labels,
            ],
            cwd=root,
            env=env,
            capture_output=False,
        )

        pr_number = _get_pr_number(root, branch, env)
        _append_promotion_event(
            root=root,
            state_repo_dir=state_repo_dir,
            campaign_id=cfg.campaign_id,
            seed_id=acc.seed_id,
            event_type="promotion_applied",
            run_id=acc.run_id,
            reason=f"created pr={pr_number} branch={branch}",
        )
        created += 1

        # Switch back to base for the next promotion.
        _checkout_base(root, args.base)

    print(
        json.dumps(
            {
                "campaign_id": cfg.campaign_id,
                "created_prs": created,
            },
            ensure_ascii=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
