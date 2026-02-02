#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REQUIRED_KEYS = [
    "experiment type",
    "risk_level",
    "max_diff_size",
    "roi",
    "total stake",
    "n_bets",
    "test period",
    "max drawdown",
    "roi definition",
    "rolling",
    "pooled vs step14 sign mismatch",
    "preferred roi for decisions",
]

ROLLING_KEYS = [
    "design window",
    "eval window",
    "paired delta vs baseline",
]

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


def validate_log(path):
    text = Path(path).read_text(encoding="utf-8")
    fields = parse_fields(text)
    errors = []

    for key in REQUIRED_KEYS:
        if key not in fields:
            errors.append(f"Missing field '{key}'")

    exp_type = fields.get("experiment type", "experiment").lower()
    if exp_type not in {"experiment", "infra"}:
        errors.append("Experiment type must be 'experiment' or 'infra'")

    rolling = fields.get("rolling", "").lower()
    if rolling not in {"yes", "no"}:
        errors.append("Rolling must be 'yes' or 'no'")

    if rolling == "yes":
        for key in ROLLING_KEYS:
            if key not in fields:
                errors.append(f"Missing rolling field '{key}'")
            elif is_na(fields[key]):
                errors.append(f"Rolling field '{key}' cannot be N/A when Rolling is yes")
    else:
        for key in ROLLING_KEYS:
            if key not in fields:
                errors.append(f"Missing rolling field '{key}'")

    if fields.get("pooled vs step14 sign mismatch", "").lower() not in {"yes", "no"}:
        errors.append("Pooled vs step14 sign mismatch must be 'yes' or 'no'")

    if fields.get("preferred roi for decisions", "").lower() not in {"step14", "pooled"}:
        errors.append("Preferred ROI for decisions must be 'step14' or 'pooled'")

    roi_def = fields.get("roi definition", "")
    if "roi = profit / stake" not in roi_def.lower() or "profit = return - stake" not in roi_def.lower():
        errors.append("ROI definition must include 'ROI = profit / stake' and 'profit = return - stake'")

    # Metrics fields validation
    if exp_type == "infra":
        for key in ["roi", "total stake", "n_bets", "test period", "max drawdown"]:
            if key in fields and not is_na(fields[key]):
                errors.append(f"{key} should be N/A for infra experiments")
    else:
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
        print("No changes detected; skipping experiment log checks.")
        return 0

    code_changes = [f for f in files if is_code_change(f)]
    exp_logs = [f for f in files if f.startswith("docs/experiments/") and f.endswith(".md") and not f.endswith("_template.md")]

    if code_changes and not exp_logs:
        print("Experiment log required for code changes but none found.", file=sys.stderr)
        print("Changed files:", file=sys.stderr)
        for f in code_changes:
            print(f"- {f}", file=sys.stderr)
        return 1

    failures = []
    for log_path in exp_logs:
        full_path = root / log_path
        if not full_path.exists():
            failures.append(f"Missing experiment log file: {log_path}")
            continue
        errors = validate_log(full_path)
        if errors:
            failures.append(f"{log_path}: " + "; ".join(errors))

    if failures:
        print("Experiment log validation failed:", file=sys.stderr)
        for err in failures:
            print(f"- {err}", file=sys.stderr)
        return 1

    print("Experiment log validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
