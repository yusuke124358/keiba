#!/usr/bin/env python3
import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path


def repo_root() -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return Path(out)


def gh_has_needs_human(root: Path, label: str) -> bool:
    result = subprocess.run(
        ["gh", "pr", "list", "--label", label, "--state", "open", "--limit", "1"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return bool(result.stdout.strip())


def run(cmd, root: Path) -> None:
    result = subprocess.run(cmd, cwd=root)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")


def git_ahead_count(root: Path, base_ref: str = "origin/main") -> int:
    result = subprocess.run(
        ["git", "rev-list", "--count", f"{base_ref}..HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return int(result.stdout.strip())
    except Exception:
        return 0


def write_loop_artifacts(root: Path, title: str, labels: str) -> Path:
    out_dir = root / "artifacts" / "agent"
    out_dir.mkdir(parents=True, exist_ok=True)
    body_path = out_dir / "loop_pr_body.md"
    body = [
        "# Scientist Loop PR",
        "",
        f"- Generated at: {datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')}",
        f"- Title: {title}",
        "",
        "## Repro",
        "- `make ci`",
        "",
        "## Notes",
        "- This PR was created by the scientist loop.",
    ]
    body_path.write_text("\n".join(body), encoding="utf-8")
    (out_dir / "loop_labels.txt").write_text(labels, encoding="utf-8")
    return body_path


def load_experiment_decision(root: Path, run_id: str) -> str:
    path = root / "experiments" / "runs" / f"{run_id}.json"
    if not path.exists():
        return ""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return ""
        decision = data.get("decision")
        if isinstance(decision, dict):
            decision = decision.get("decision")
        decision = str(decision or "").strip().lower()
        if decision:
            return decision

        # Backward-compat for legacy results that only have status.
        status = str(data.get("status") or "").strip().lower()
        if status == "pass":
            return "accept"
        if status == "fail":
            return "reject"
        if status in {"needs-human", "needs_human"}:
            return "needs-human"
        if status == "inconclusive":
            return "iterate"
        return ""
    except Exception:
        return ""


def write_publish_marker(
    root: Path,
    publish: bool,
    reason: str,
    run_id: str = "",
    decision: str = "",
    labels: str = "",
) -> None:
    out_dir = root / "artifacts" / "agent"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "publish": bool(publish),
        "reason": str(reason or ""),
        "run_id": str(run_id or ""),
        "decision": str(decision or ""),
        "labels": str(labels or ""),
    }
    (out_dir / "loop_publish.json").write_text(
        json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8"
    )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--once", action="store_true", help="Run at most one action.")
    p.add_argument("--needs-human-label", default="needs-human")
    args = p.parse_args()

    root = repo_root()

    if gh_has_needs_human(root, args.needs_human_label):
        print("needs-human present; skipping loop.")
        return 0

    # Step 1: checkpoint if due
    checkpoint_result = subprocess.run(
        ["python", "scripts/agent/checkpoint.py"], cwd=root
    )
    if checkpoint_result.returncode == 0:
        if git_ahead_count(root) > 0:
            labels = "checkpoint,needs-human"
            write_loop_artifacts(root, "checkpoint", labels)
            write_publish_marker(root, True, "checkpoint", labels=labels)
            print("Checkpoint created; stopping after one action.")
            return 0
    elif checkpoint_result.returncode != 2:
        raise RuntimeError(
            f"Command failed ({checkpoint_result.returncode}): python scripts/agent/checkpoint.py"
        )

    # Step 2: plan + run one experiment
    run(["python", "scripts/agent/plan_next_experiment.py"], root)
    plans = list((root / "artifacts" / "agent").glob("plan_*.json"))
    if not plans:
        print("No plan generated.")
        write_publish_marker(root, False, "no_plan")
        return 0
    plan_path = max(plans, key=lambda p: p.stat().st_mtime)
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
    except Exception:
        plan = {}
    run_id = str(plan.get("run_id") or "").strip()
    run(["python", "scripts/agent/run_experiment.py", "--plan", str(plan_path)], root)
    decision = load_experiment_decision(root, run_id) if run_id else ""
    publish_ok = decision in {"accept", "needs-human"}
    labels = "autogen,auto-fix"
    if decision == "needs-human":
        labels = "autogen,needs-human"

    write_publish_marker(
        root,
        publish_ok and git_ahead_count(root) > 0,
        "decision_gate",
        run_id=run_id,
        decision=decision,
        labels=labels,
    )

    if git_ahead_count(root) > 0 and publish_ok:
        write_loop_artifacts(root, "experiment", labels)
    print("Experiment run complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
