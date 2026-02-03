#!/usr/bin/env python3
import argparse
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
    try:
        run(["python", "scripts/agent/checkpoint.py"], root)
        if git_ahead_count(root) > 0:
            labels = "checkpoint,needs-human"
            write_loop_artifacts(root, "checkpoint", labels)
            print("Checkpoint created; stopping after one action.")
            return 0
    except Exception:
        pass

    # Step 2: plan + run one experiment
    run(["python", "scripts/agent/plan_next_experiment.py"], root)
    plans = sorted((root / "artifacts" / "agent").glob("plan_*.json"))
    if not plans:
        print("No plan generated.")
        return 0
    plan_path = plans[-1]
    run(["python", "scripts/agent/run_experiment.py", "--plan", str(plan_path)], root)
    if git_ahead_count(root) > 0:
        write_loop_artifacts(root, "experiment", "autogen,auto-fix")
    print("Experiment run complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
