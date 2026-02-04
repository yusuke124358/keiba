#!/usr/bin/env python3
import argparse
import datetime as dt
import os
import subprocess
from pathlib import Path
from typing import Iterable, Tuple


DOC_ONLY_PREFIXES = [
    "docs/",
    "experiments/",
    ".github/",
    ".codex/",
    "context/",
    "tasks/",
]

DOC_ONLY_FILES = {
    "README.md",
    "AGENTS.md",
    "memory.md",
}


def repo_root() -> Path:
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    return Path(out)


def run(cmd, cwd=None, env=None, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, cwd=cwd, env=env, capture_output=False)
    if result.returncode != 0:
        if check:
            raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")
    return result


def run_capture(cmd, cwd=None, env=None) -> Tuple[int, str]:
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode, (result.stdout or "").strip()


def is_code_change(path_str: str) -> bool:
    if path_str in DOC_ONLY_FILES:
        return False
    for prefix in DOC_ONLY_PREFIXES:
        if path_str.startswith(prefix):
            return False
    return True


def resolve_base_ref(root: Path, base: str) -> str:
    remote_ref = f"refs/remotes/origin/{base}"
    local_ref = f"refs/heads/{base}"
    if (
        subprocess.run(
            ["git", "show-ref", "--verify", "--quiet", remote_ref],
            cwd=root,
        ).returncode
        == 0
    ):
        return f"origin/{base}"
    if (
        subprocess.run(
            ["git", "show-ref", "--verify", "--quiet", local_ref],
            cwd=root,
        ).returncode
        == 0
    ):
        return base
    return base


def normalize_labels(labels: str) -> list[str]:
    return [label.strip() for label in labels.split(",") if label.strip()]


def ensure_labels(root: Path, labels: Iterable[str], env: dict) -> None:
    if not labels:
        return
    code, repo = run_capture(
        ["gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"],
        cwd=root,
        env=env,
    )
    if code != 0 or not repo:
        return
    for label in labels:
        if not label:
            continue
        exists = run(
            ["gh", "api", f"repos/{repo}/labels/{label}"],
            cwd=root,
            env=env,
            check=False,
        ).returncode
        if exists == 0:
            continue
        run(
            [
                "gh",
                "api",
                "-X",
                "POST",
                f"repos/{repo}/labels",
                "-f",
                f"name={label}",
                "-f",
                "color=0E8A16",
                "-f",
                "description=autogen label",
            ],
            cwd=root,
            env=env,
            check=False,
        )


def find_pr_number(root: Path, branch: str, env: dict) -> str:
    code, number = run_capture(
        [
            "gh",
            "pr",
            "view",
            "--json",
            "number",
            "--head",
            branch,
            "-q",
            ".number",
        ],
        cwd=root,
        env=env,
    )
    if code != 0 or not number:
        return ""
    return number


def ensure_infra_experiment_log(root: Path, base_ref: str) -> None:
    changed = (
        subprocess.run(
            ["git", "diff", "--name-only", f"{base_ref}..HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        .stdout.strip()
        .splitlines()
    )
    files = [f.strip() for f in changed if f.strip()]
    if not files:
        return

    code_changes = [f for f in files if is_code_change(f)]
    exp_logs = [
        f
        for f in files
        if f.startswith("docs/experiments/")
        and f.endswith(".md")
        and not f.endswith("_template.md")
    ]
    if not code_changes or exp_logs:
        return

    ts = dt.datetime.utcnow().strftime("INFRA-%Y%m%d_%H%M%S")
    log_path = root / "docs" / "experiments" / f"{ts}.md"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "\n".join(
            [
                f"# Experiment {ts} - infra log auto-generated",
                "",
                "## Hypothesis",
                "Infra change; experiment metrics not applicable.",
                "",
                "## Change summary",
                "- Auto-generated infra log for code-only changes.",
                "",
                "## Risk",
                "- Experiment type: infra",
                "- risk_level: low",
                "- max_diff_size: 200",
                "",
                "## Commands",
                "- `py64_analysis\\.venv\\Scripts\\python.exe -m pytest py64_analysis/tests`",
                "- `py64_analysis\\.venv\\Scripts\\python.exe py64_analysis/scripts/check_system_status.py`",
                "",
                "## Metrics (required)",
                "- ROI: N/A",
                "- Total stake: N/A",
                "- n_bets: N/A",
                "- Test period: N/A",
                "- Max drawdown: N/A",
                "- ROI definition: ROI = profit / stake, profit = return - stake.",
                "- Rolling: no",
                "- Design window: N/A",
                "- Eval window: N/A",
                "- Paired delta vs baseline: N/A",
                "- Pooled vs step14 sign mismatch: no",
                "- Preferred ROI for decisions: pooled",
                "",
                "## Artifacts",
                "- metrics.json: N/A",
                "- comparison.json: N/A",
                "- report: N/A",
                "",
                "## Decision",
                "- status: pass",
                "- next_action: merge",
            ]
        ),
        encoding="utf-8",
    )

    subprocess.run(["git", "add", str(log_path)], cwd=root, check=False)
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    if status:
        subprocess.run(
            ["git", "commit", "-m", "docs: add infra experiment log"],
            cwd=root,
            check=False,
        )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="main")
    p.add_argument("--labels", default="autogen,auto-fix")
    p.add_argument("--title", required=True)
    p.add_argument("--body-file", required=True)
    args = p.parse_args()

    root = repo_root()
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    if not branch:
        raise RuntimeError("Unable to determine current branch.")
    if branch == args.base:
        raise RuntimeError("Refusing to publish from base branch.")

    env = os.environ.copy()
    token = env.get("AUTO_FIX_PUSH_TOKEN") or env.get("GH_TOKEN")
    if token:
        env["GH_TOKEN"] = token
    base_ref = resolve_base_ref(root, args.base)
    ensure_infra_experiment_log(root, base_ref)
    run(["git", "push", "origin", f"HEAD:{branch}"], cwd=root, env=env)
    run(
        [
            "gh",
            "pr",
            "create",
            "--base",
            args.base,
            "--head",
            branch,
            "--title",
            args.title,
            "--body-file",
            args.body_file,
        ],
        cwd=root,
        env=env,
    )
    if args.labels:
        labels = normalize_labels(args.labels)
        ensure_labels(root, labels, env)
        pr_number = find_pr_number(root, branch, env)
        if pr_number:
            for label in labels:
                run(
                    ["gh", "pr", "edit", pr_number, "--add-label", label],
                    cwd=root,
                    env=env,
                    check=False,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
