#!/usr/bin/env python3
import argparse
import re
import subprocess
import sys
from pathlib import Path

METRIC_KEYS = ["roi", "total stake", "n_bets", "test period", "max drawdown"]

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


def run(cmd, cwd=None, check=True):
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if check and result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\n{result.stderr.strip()}"
        )
    return result.stdout.strip()


def repo_root():
    return Path(run(["git", "rev-parse", "--show-toplevel"]))


def changed_files(base, head, root):
    out = run(["git", "diff", "--name-only", f"{base}...{head}"], cwd=root)
    return [line.strip() for line in out.splitlines() if line.strip()]


def is_code_change(path_str):
    if path_str in DOC_ONLY_FILES:
        return False
    for prefix in DOC_ONLY_PREFIXES:
        if path_str.startswith(prefix):
            return False
    return True


def parse_fields(text):
    fields = {}
    for line in text.splitlines():
        m = re.match(r"^\s*-\s*([^:]+):\s*(.+)\s*$", line)
        if m:
            key = m.group(1).strip().lower()
            fields[key] = m.group(2).strip()
    return fields


def is_na(value):
    return value.lower().startswith("n/a")


def validate_eval_fields(path):
    text = Path(path).read_text(encoding="utf-8")
    fields = parse_fields(text)
    errors = []

    exp_type = fields.get("experiment type", "experiment").lower()
    if exp_type == "infra":
        return errors

    missing = [k for k in METRIC_KEYS if k not in fields]
    if missing:
        errors.append(f"Missing metric fields: {', '.join(missing)}")
        return errors

    for key in METRIC_KEYS:
        if is_na(fields[key]):
            errors.append(f"{key} must not be N/A for experiment results")

    test_period = fields.get("test period", "")
    if not re.match(r"^\d{4}-\d{2}-\d{2}\s+to\s+\d{4}-\d{2}-\d{2}$", test_period):
        errors.append("Test period must be 'YYYY-MM-DD to YYYY-MM-DD'")

    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="origin/main")
    parser.add_argument("--head", default="HEAD")
    args = parser.parse_args()

    root = repo_root()
    files = changed_files(args.base, args.head, root)
    if not files:
        print("No changes detected; skipping evaluation gate.")
        return 0

    code_changes = [f for f in files if is_code_change(f)]
    exp_logs = [
        f
        for f in files
        if f.startswith("docs/experiments/")
        and f.endswith(".md")
        and not f.endswith("_template.md")
    ]

    if not code_changes:
        print("No code changes; skipping evaluation gate.")
        return 0

    failures = []
    for log_path in exp_logs:
        full_path = root / log_path
        if not full_path.exists():
            failures.append(f"Missing experiment log file: {log_path}")
            continue
        errors = validate_eval_fields(full_path)
        if errors:
            failures.append(f"{log_path}: " + "; ".join(errors))

    if failures:
        print("Evaluation gate failed:", file=sys.stderr)
        for err in failures:
            print(f"- {err}", file=sys.stderr)
        return 1

    print("Evaluation gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
