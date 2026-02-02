#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except Exception as exc:
    print("PyYAML is required to run the agent loop. Install with: pip install pyyaml", file=sys.stderr)
    raise


def run(cmd, cwd=None, check=True, capture_output=True, text=True):
    result = subprocess.run(cmd, cwd=cwd, check=False, capture_output=capture_output, text=text)
    if check and result.returncode != 0:
        stderr = result.stderr.strip() if result.stderr else ""
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}\n{stderr}")
    return result


def repo_root():
    out = run(["git", "rev-parse", "--show-toplevel"]).stdout.strip()
    return Path(out)


def slugify(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "change"


def ensure_clean(root):
    status = run(["git", "status", "--porcelain"], cwd=root).stdout.strip()
    if status:
        raise RuntimeError("Working tree not clean. Commit or stash changes before running the loop.")


def git_fetch_checkout(root, remote, base_branch):
    run(["git", "fetch", remote], cwd=root)
    run(["git", "checkout", base_branch], cwd=root)
    run(["git", "pull", "--ff-only", remote, base_branch], cwd=root)


def load_backlog(path):
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "items" not in data:
        raise RuntimeError("Invalid backlog format: expected top-level 'items'")
    return data


def pick_item(items, pick_id=None):
    if pick_id:
        for item in items:
            if item.get("id") == pick_id:
                return item
        return None
    for item in items:
        status = str(item.get("status", "todo")).lower()
        if status in {"todo", "ready"}:
            return item
    return None


def update_item(item, branch):
    item["status"] = "in_progress"
    item["picked_at"] = dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    item["branch"] = branch


def save_backlog(path, data):
    Path(path).write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=False), encoding="utf-8")


def ensure_experiment_log(root, exp_id, title, risk_level, max_diff_size):
    log_path = root / "docs" / "experiments" / f"{exp_id}.md"
    if log_path.exists():
        return log_path

    template_path = root / "docs" / "experiments" / "_template.md"
    if not template_path.exists():
        raise RuntimeError("Missing docs/experiments/_template.md")

    text = template_path.read_text(encoding="utf-8")
    text = text.replace("<id>", exp_id)
    text = text.replace("<title>", title)
    text = text.replace("<low|medium|high>", str(risk_level))
    text = text.replace("<int>", str(max_diff_size))
    text = text.replace("experiment|infra", "experiment")
    log_path.write_text(text, encoding="utf-8")
    return log_path


def render_prompt(template_path, item):
    text = Path(template_path).read_text(encoding="utf-8")
    def render_list(value):
        if not value:
            return "- (none)"
        if isinstance(value, str):
            return f"- {value}"
        return "\n".join(f"- {v}" for v in value)

    replacements = {
        "{{id}}": item.get("id", ""),
        "{{title}}": item.get("title", ""),
        "{{hypothesis}}": item.get("hypothesis", ""),
        "{{risk_level}}": item.get("risk_level", ""),
        "{{max_diff_size}}": str(item.get("max_diff_size", "")),
        "{{change_scope}}": render_list(item.get("change_scope")),
        "{{acceptance_criteria}}": render_list(item.get("acceptance_criteria")),
        "{{metrics}}": render_list(item.get("metrics")),
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


def find_codex_bin():
    if os.name == "nt":
        for name in ("codex.cmd", "codex.exe", "codex"):
            path = shutil.which(name)
            if path:
                return path
    return shutil.which("codex")


def run_codex(root, prompt_text, schema_path, output_last_message, profile, log_path, codex_bin):
    cmd = [codex_bin, "exec", "--profile", profile, "--full-auto",
           "--output-schema", str(schema_path),
           "--output-last-message", str(output_last_message)]

    with open(log_path, "w", encoding="utf-8") as log:
        result = subprocess.run(
            cmd + [prompt_text],
            cwd=root,
            stdout=log,
            stderr=log,
            text=True,
        )
    if result.returncode != 0:
        raise RuntimeError(f"codex exec failed with code {result.returncode}. See {log_path}")


def diff_size(root):
    run(["git", "add", "-A"], cwd=root)
    out = run(["git", "diff", "--cached", "--numstat"], cwd=root).stdout.strip()
    added = deleted = 0
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        try:
            a = int(parts[0]) if parts[0].isdigit() else 0
            d = int(parts[1]) if parts[1].isdigit() else 0
        except ValueError:
            a = d = 0
        added += a
        deleted += d
    return added + deleted


def run_verify(root):
    if os.name == "nt":
        cmd = ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(root / "scripts" / "verify.ps1")]
    else:
        cmd = ["bash", str(root / "scripts" / "verify.sh")]
    result = subprocess.run(cmd, cwd=root)
    if result.returncode != 0:
        raise RuntimeError(f"verify failed with code {result.returncode}")


def load_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return None


def render_pr_body(item, codex_output):
    summary = (codex_output or {}).get("summary", "(summary missing)")
    change_summary = (codex_output or {}).get("change_summary", [])
    tests_run = (codex_output or {}).get("tests_run", [])
    risks = (codex_output or {}).get("risks", [])
    metrics = (codex_output or {}).get("metrics", {})
    artifacts = (codex_output or {}).get("artifacts", {})

    lines = []
    lines.append("## Summary")
    lines.append(summary)
    lines.append("")

    if change_summary:
        lines.append("## Changes")
        for item_line in change_summary:
            lines.append(f"- {item_line}")
        lines.append("")

    if tests_run:
        lines.append("## Tests")
        for test in tests_run:
            lines.append(f"- {test}")
        lines.append("")

    lines.append("## Experiment")
    lines.append(f"- id: {item.get('id')}")
    lines.append(f"- risk_level: {item.get('risk_level')}")
    lines.append(f"- log: docs/experiments/{item.get('id')}.md")
    lines.append("")

    if metrics:
        lines.append("## Metrics")
        for key, value in metrics.items():
            lines.append(f"- {key}: {value}")
        lines.append("")

    if artifacts:
        lines.append("## Artifacts")
        for key, value in artifacts.items():
            lines.append(f"- {key}: {value}")
        lines.append("")

    if risks:
        lines.append("## Risks")
        for risk in risks:
            lines.append(f"- {risk}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def gh_json(args, root):
    result = run(["gh"] + args, cwd=root, capture_output=True)
    return json.loads(result.stdout)


def ensure_labels(root, labels):
    for label in labels:
        try:
            run(["gh", "label", "create", label["name"], "--color", label["color"],
                 "--description", label["description"], "--force"], cwd=root)
        except Exception:
            pass


def pr_checks_green(root, pr_number):
    try:
        data = gh_json(["pr", "view", str(pr_number), "--json", "statusCheckRollup"], root)
    except Exception:
        return False, ["Unable to query checks"]

    rollup = data.get("statusCheckRollup", []) or []
    bad = []
    for check in rollup:
        state = (check.get("state") or check.get("conclusion") or "").upper()
        name = check.get("name", "unknown")
        if state in {"SUCCESS", "NEUTRAL", "SKIPPED"}:
            continue
        if not state:
            continue
        bad.append(f"{name}:{state}")
    return len(bad) == 0, bad


def pr_has_changes_requested(root, pr_number):
    try:
        data = gh_json(["pr", "view", str(pr_number), "--json", "reviewDecision,reviews,comments"], root)
    except Exception:
        return False, []

    reasons = []
    if data.get("reviewDecision") == "CHANGES_REQUESTED":
        reasons.append("reviewDecision: CHANGES_REQUESTED")

    for review in data.get("reviews", []) or []:
        if review.get("state") == "CHANGES_REQUESTED":
            author = (review.get("author") or {}).get("login", "unknown")
            reasons.append(f"review:{author}: CHANGES_REQUESTED")

    return len(reasons) > 0, reasons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backlog", default="experiments/backlog.yml")
    parser.add_argument("--id", default="")
    parser.add_argument("--base-branch", default="main")
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--profile", default="agent_loop")
    parser.add_argument("--schema", default="scripts/agent/output_schema.json")
    parser.add_argument("--prompt", default="scripts/agent/prompt.md")
    parser.add_argument("--output-dir", default="artifacts/agent")
    parser.add_argument("--skip-codex", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-pr", action="store_true")
    parser.add_argument("--skip-review-comment", action="store_true")
    parser.add_argument("--skip-auto-merge", action="store_true")
    parser.add_argument("--fix-reviews", action="store_true")
    parser.add_argument("--max-fix-rounds", type=int, default=2)
    args = parser.parse_args()

    root = repo_root()
    os.chdir(root)

    ensure_clean(root)
    git_fetch_checkout(root, args.remote, args.base_branch)

    backlog_path = root / args.backlog
    data = load_backlog(backlog_path)
    item = pick_item(data.get("items", []), args.id or None)
    if not item:
        print("No backlog items available.")
        return 0

    exp_id = item.get("id")
    title = item.get("title", "")
    risk_level = item.get("risk_level", "medium")
    max_diff_size = int(item.get("max_diff_size", 200))

    branch = f"agent/{exp_id}-{slugify(title)}"
    run(["git", "checkout", "-b", branch], cwd=root)

    update_item(item, branch)
    save_backlog(backlog_path, data)

    ensure_experiment_log(root, exp_id, title, risk_level, max_diff_size)

    ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"{exp_id}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    codex_log = out_dir / "codex_exec.log"
    codex_last = out_dir / "codex_last_message.json"

    if not args.skip_codex:
        codex_bin = find_codex_bin()
        if not codex_bin:
            raise RuntimeError("codex CLI not found in PATH")
        prompt_text = render_prompt(root / args.prompt, item)
        run_codex(root, prompt_text, root / args.schema, codex_last, args.profile, codex_log, codex_bin)

    total_diff = diff_size(root)
    if total_diff > max_diff_size:
        run(["git", "reset"], cwd=root, check=False)
        raise RuntimeError(f"Diff size {total_diff} exceeds max_diff_size {max_diff_size}")

    if not args.skip_tests:
        run_verify(root)

    run(["git", "add", "-A"], cwd=root)
    status = run(["git", "status", "--porcelain"], cwd=root).stdout.strip()
    if not status:
        print("No changes to commit.")
        return 0

    commit_msg = f"agent: {exp_id} {slugify(title)}"
    run(["git", "commit", "-m", commit_msg], cwd=root)
    run(["git", "push", "-u", args.remote, branch], cwd=root)

    if args.skip_pr:
        print("PR creation skipped.")
        return 0

    codex_output = load_json(codex_last)
    pr_body = (codex_output or {}).get("pr_body") or render_pr_body(item, codex_output)
    pr_body_path = out_dir / "pr_body.md"
    pr_body_path.write_text(pr_body, encoding="utf-8")

    pr_title = f"{exp_id}: {title}"
    run(["gh", "auth", "status"], cwd=root)
    run(["gh", "pr", "create", "--base", args.base_branch, "--head", branch,
         "--title", pr_title, "--body-file", str(pr_body_path)], cwd=root)

    pr_data = gh_json(["pr", "view", "--json", "number,url", "--head", branch], root)
    pr_number = pr_data.get("number")
    pr_url = pr_data.get("url")

    ensure_labels(root, [
        {"name": "agent-loop", "color": "0e8a16", "description": "Automated agent loop PR"},
        {"name": "risk:low", "color": "0e8a16", "description": "Low risk"},
        {"name": "risk:medium", "color": "fbca04", "description": "Medium risk"},
        {"name": "risk:high", "color": "d93f0b", "description": "High risk"},
    ])

    run(["gh", "pr", "edit", str(pr_number), "--add-label", "agent-loop"], cwd=root)
    run(["gh", "pr", "edit", str(pr_number), "--add-label", f"risk:{risk_level}"], cwd=root)

    if not args.skip_review_comment:
        run(["gh", "pr", "comment", str(pr_number), "--body", "@codex review"], cwd=root)

    if args.fix_reviews:
        for _ in range(args.max_fix_rounds):
            has_changes, reasons = pr_has_changes_requested(root, pr_number)
            checks_ok, bad_checks = pr_checks_green(root, pr_number)
            if not has_changes and checks_ok:
                break
            issues = reasons + bad_checks
            if not issues:
                break
            prompt = (
                "Fix PR review issues and failing checks. "
                f"PR: {pr_url}\nIssues:\n" + "\n".join(f"- {i}" for i in issues)
                + "\n\nRequirements:\n"
                + "- Keep the change scoped to the current hypothesis.\n"
                + f"- Update docs/experiments/{exp_id}.md if results change.\n"
                + "- Output ONLY valid JSON matching scripts/agent/output_schema.json.\n"
            )
            fix_log = out_dir / f"codex_fix_{dt.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log"
            codex_bin = find_codex_bin()
            if not codex_bin:
                raise RuntimeError("codex CLI not found in PATH")
            run_codex(root, prompt, root / args.schema, codex_last, args.profile, fix_log, codex_bin)
            run_verify(root)
            run(["git", "add", "-A"], cwd=root)
            run(["git", "commit", "-m", f"agent: fix {exp_id}"], cwd=root)
            run(["git", "push", args.remote, branch], cwd=root)

    if risk_level == "high":
        print("risk_level=high; auto-merge disabled.")
        return 0

    if not args.skip_auto_merge:
        checks_ok, bad_checks = pr_checks_green(root, pr_number)
        if checks_ok:
            run(["gh", "pr", "merge", str(pr_number), "--auto", "--squash"], cwd=root)
        else:
            print("Checks not green; auto-merge not enabled:")
            for bad in bad_checks:
                print(f"- {bad}")

    print(f"PR created: {pr_url}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
