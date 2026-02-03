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
except Exception:
    print(
        "PyYAML is required to run propose_pr. Install with: pip install pyyaml",
        file=sys.stderr,
    )
    raise

from issue_id import build_issue_id


PROMPTS_DIR = Path("prompts")


def run(cmd, cwd=None, check=True, capture_output=True, text=True, env=None):
    result = subprocess.run(
        cmd,
        cwd=cwd,
        check=False,
        capture_output=capture_output,
        text=text,
        env=env,
    )
    if check and result.returncode != 0:
        stderr = result.stderr.strip() if result.stderr else ""
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\n{stderr}"
        )
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
        raise RuntimeError(
            "Working tree not clean. Commit or stash changes before running propose_pr."
        )


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
    Path(path).write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=False), encoding="utf-8"
    )


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


def ensure_codex_ready(codex_bin):
    result = subprocess.run(
        [codex_bin, "login", "status"], capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(
            "codex login status failed. Run: codex login --device-auth on the runner."
        )


def codex_help(codex_bin: str) -> str:
    result = subprocess.run(
        [codex_bin, "exec", "--help"], capture_output=True, text=True
    )
    if result.returncode != 0:
        return ""
    return (result.stdout or "") + (result.stderr or "")


def flag_supported(help_text: str, flag: str) -> bool:
    return flag in help_text


def run_codex(prompt_text, schema_path, output_path, log_path, profile, codex_bin):
    help_text = codex_help(codex_bin)
    cmd = [codex_bin, "exec"]
    if not help_text or flag_supported(help_text, "--profile"):
        cmd.extend(["--profile", profile])
    if flag_supported(help_text, "--ask-for-approval"):
        cmd.extend(["--ask-for-approval", "never"])
    if flag_supported(help_text, "--sandbox"):
        cmd.extend(["--sandbox", "workspace-write"])
    if flag_supported(help_text, "--output-schema"):
        cmd.extend(["--output-schema", str(schema_path)])
    supports_output_last = flag_supported(help_text, "--output-last-message")
    if supports_output_last:
        cmd.extend(["--output-last-message", str(output_path)])

    if supports_output_last:
        with open(log_path, "w", encoding="utf-8") as log:
            result = subprocess.run(
                cmd + [prompt_text], stdout=log, stderr=log, text=True
            )
    else:
        result = subprocess.run(cmd + [prompt_text], capture_output=True, text=True)
        with open(log_path, "w", encoding="utf-8") as log:
            log.write(result.stdout or "")
            if result.stderr:
                log.write("\n")
                log.write(result.stderr)

    if result.returncode != 0:
        tail = ""
        try:
            text = Path(log_path).read_text(encoding="utf-8")
            tail = text[-4000:] if len(text) > 4000 else text
        except Exception:
            tail = ""
        message = f"codex exec failed with code {result.returncode}. See {log_path}"
        if tail:
            message = f"{message}\n--- codex log tail ---\n{tail}"
        raise RuntimeError(message)

    if not supports_output_last:
        output_path.write_text(result.stdout or "", encoding="utf-8")


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


def run_make_ci(root, base_ref):
    env = os.environ.copy()
    env["VERIFY_BASE"] = base_ref
    # Ensure pytest temp dir is writable on Windows runners.
    tmp_root = Path(root) / "tmp" / "pytest"
    tmp_root.mkdir(parents=True, exist_ok=True)
    run_id = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    tmp_session = tmp_root / f"pytest_{run_id}"
    tmp_session.mkdir(parents=True, exist_ok=True)
    env.setdefault("TMP", str(tmp_session))
    env.setdefault("TEMP", str(tmp_session))
    env.setdefault("TMPDIR", str(tmp_session))
    env.setdefault("PYTEST_TMPDIR", str(tmp_session))
    if shutil.which("make"):
        result = subprocess.run(["make", "ci"], cwd=root, env=env)
        return result.returncode
    ci_ps1 = Path(root) / "scripts" / "ci.ps1"
    verify_ps1 = Path(root) / "scripts" / "verify.ps1"
    if ci_ps1.exists():
        result = subprocess.run(
            ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(ci_ps1)],
            cwd=root,
            env=env,
        )
        return result.returncode
    if verify_ps1.exists():
        result = subprocess.run(
            ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(verify_ps1)],
            cwd=root,
            env=env,
        )
        return result.returncode
    raise RuntimeError("make not found and no scripts/ci.ps1 or scripts/verify.ps1")


def normalize_review_items(review_items_path):
    data = json.loads(review_items_path.read_text(encoding="utf-8"))
    issues = []
    seen = set()
    for issue in data.get("issues", []):
        issue_id = build_issue_id(
            issue.get("source", ""),
            issue.get("file", "") or "",
            issue.get("line", "") or "",
            issue.get("message", ""),
        )
        issue["id"] = issue_id
        if issue_id in seen:
            continue
        seen.add(issue_id)
        issues.append(issue)
    data["issues"] = issues
    review_items_path.write_text(
        json.dumps(data, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )
    return data


def build_diff(root, base_ref, out_path):
    diff = run(["git", "diff", f"{base_ref}...HEAD"], cwd=root).stdout
    out_path.write_text(diff, encoding="utf-8")


def load_prompt(path):
    if not path.exists():
        raise RuntimeError(f"Missing prompt: {path}")
    return path.read_text(encoding="utf-8").strip()


def build_review_prompt(diff_path):
    base = load_prompt(PROMPTS_DIR / "reviewer.md")
    return "\n".join(
        [
            base,
            "",
            f"Diff file: {diff_path}",
            "Review the diff for correctness, risks, and missing tests.",
            "Include file/line when possible.",
        ]
    )


def build_manager_prompt(review_items_path):
    base = load_prompt(PROMPTS_DIR / "manager.md")
    review_text = ""
    if review_items_path.exists():
        review_text = review_items_path.read_text(encoding="utf-8").strip()
    return "\n".join(
        [
            base,
            "",
            f"Review items JSON path: {review_items_path}",
            "Review items JSON content:",
            review_text or "{}",
            "Apply business rules from AGENTS.md.",
        ]
    )


def build_fixer_prompt(manager_path, review_items_path):
    base = load_prompt(PROMPTS_DIR / "fixer.md")
    manager_text = ""
    review_text = ""
    if manager_path.exists():
        manager_text = manager_path.read_text(encoding="utf-8").strip()
    if review_items_path.exists():
        review_text = review_items_path.read_text(encoding="utf-8").strip()
    return "\n".join(
        [
            base,
            "",
            f"Manager decisions path: {manager_path}",
            "Manager decisions JSON:",
            manager_text or "{}",
            f"Review items path: {review_items_path}",
            "Review items JSON:",
            review_text or "{}",
        ]
    )


def load_labels():
    config_path = Path("config/auto_fix.yml")
    if not config_path.exists():
        return {
            "auto_fix": "auto-fix",
            "needs_human": "needs-human",
            "autogen": "autogen",
        }
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    labels = data.get("labels") or {}
    return {
        "auto_fix": labels.get("auto_fix", "auto-fix"),
        "needs_human": labels.get("needs_human", "needs-human"),
        "autogen": labels.get("autogen", "autogen"),
    }


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
    parser.add_argument(
        "--publisher", choices=["none", "local", "actions", "token"], default="none"
    )
    parser.add_argument("--once", action="store_true")
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
    run(["git", "checkout", "-B", branch], cwd=root)

    update_item(item, branch)
    save_backlog(backlog_path, data)

    ensure_experiment_log(root, exp_id, title, risk_level, max_diff_size)

    ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"propose_{exp_id}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    codex_bin = find_codex_bin()
    if not codex_bin:
        raise RuntimeError("codex CLI not found in PATH")
    ensure_codex_ready(codex_bin)

    implement_output = out_dir / "implement_output.json"
    implement_log = out_dir / "implement_codex.log"
    prompt_text = render_prompt(root / args.prompt, item)
    run_codex(
        prompt_text,
        root / args.schema,
        implement_output,
        implement_log,
        args.profile,
        codex_bin,
    )

    total_diff = diff_size(root)
    if total_diff > max_diff_size:
        run(["git", "reset"], cwd=root, check=False)
        raise RuntimeError(
            f"Diff size {total_diff} exceeds max_diff_size {max_diff_size}"
        )

    diff_path = out_dir / "diff.txt"
    build_diff(root, f"origin/{args.base_branch}", diff_path)

    review_items_path = out_dir / "review_items.json"
    review_log = out_dir / "review_codex.log"
    review_prompt = build_review_prompt(diff_path)
    run_codex(
        review_prompt,
        root / "schemas/agent/review_items.schema.json",
        review_items_path,
        review_log,
        args.profile,
        codex_bin,
    )
    normalize_review_items(review_items_path)

    manager_path = out_dir / "manager_decision.json"
    manager_log = out_dir / "manager_codex.log"
    manager_prompt = build_manager_prompt(review_items_path=review_items_path)
    run_codex(
        manager_prompt,
        root / "schemas/agent/manager_decision.schema.json",
        manager_path,
        manager_log,
        args.profile,
        codex_bin,
    )

    fixer_report_path = out_dir / "fixer_report.json"
    fixer_log = out_dir / "fixer_codex.log"
    fixer_prompt = build_fixer_prompt(manager_path, review_items_path)
    run_codex(
        fixer_prompt,
        root / "schemas/agent/fixer_report.schema.json",
        fixer_report_path,
        fixer_log,
        args.profile,
        codex_bin,
    )

    run(["git", "fetch", "origin", args.base_branch], cwd=root)
    exit_code = run_make_ci(root, f"origin/{args.base_branch}")
    fixer_report = json.loads(fixer_report_path.read_text(encoding="utf-8"))
    fixer_report.setdefault("tests", [])
    fixer_report["tests"].append("make ci")
    if exit_code != 0:
        fixer_report["status"] = "failed"
        fixer_report.setdefault("failures", []).append("make ci failed")
        fixer_report["summary"] = "make ci failed"
        fixer_report["needs_human"] = True
        fixer_report_path.write_text(
            json.dumps(fixer_report, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        raise RuntimeError("make ci failed")
    fixer_report["status"] = "success"
    fixer_report["needs_human"] = False
    fixer_report_path.write_text(
        json.dumps(fixer_report, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )

    run(["git", "add", "-A"], cwd=root)
    status = run(["git", "status", "--porcelain"], cwd=root).stdout.strip()
    if not status:
        print("No changes to commit.")
        return 0

    run(["git", "config", "user.email", "autogen@users.noreply.github.com"], cwd=root)
    run(["git", "config", "user.name", "autogen-bot"], cwd=root)
    commit_msg = f"autogen: {exp_id} {slugify(title)}"
    run(["git", "commit", "-m", commit_msg], cwd=root)

    pr_body = ""
    try:
        implement_data = json.loads(implement_output.read_text(encoding="utf-8"))
        pr_body = implement_data.get("pr_body", "")
    except Exception:
        pr_body = ""
    if not pr_body:
        pr_body = f"Auto-generated PR for {exp_id}: {title}"

    labels = load_labels()
    label_str = ",".join([labels["autogen"], labels["auto_fix"]])

    pr_body_path = out_dir / "pr_body.md"
    pr_body_path.write_text(pr_body, encoding="utf-8")

    if args.publisher == "local":
        if os.name == "nt":
            publisher = root / "scripts" / "publish" / "local_push.ps1"
            run(
                [
                    "powershell",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(publisher),
                    "-BaseBranch",
                    args.base_branch,
                    "-PrTitle",
                    f"autogen: {exp_id} {title}",
                    "-PrBodyFile",
                    str(pr_body_path),
                    "-Labels",
                    label_str,
                ],
                cwd=root,
            )
        else:
            env = os.environ.copy()
            env["BASE_BRANCH"] = args.base_branch
            env["PR_TITLE"] = f"autogen: {exp_id} {title}"
            env["PR_BODY_FILE"] = str(pr_body_path)
            env["PR_LABELS"] = label_str
            publisher = root / "scripts" / "publish" / "local_push.sh"
            run(["bash", str(publisher)], cwd=root, env=env)
    elif args.publisher == "token":
        token = os.environ.get("AUTO_FIX_PUSH_TOKEN")
        if not token:
            raise RuntimeError("AUTO_FIX_PUSH_TOKEN is required for token publisher.")

        remote_url = run(
            ["git", "remote", "get-url", "origin"], cwd=root
        ).stdout.strip()
        if remote_url.startswith("git@"):
            match = re.match(r"git@github.com:(.+?/.+?)\\.git", remote_url)
            repo = match.group(1) if match else ""
            https_url = f"https://github.com/{repo}.git" if repo else remote_url
        else:
            https_url = remote_url
        push_url = https_url.replace("https://", f"https://x-access-token:{token}@")
        run(["git", "remote", "set-url", "--push", "origin", push_url], cwd=root)
        run(["git", "push", "origin", f"HEAD:{branch}"], cwd=root)

        env = os.environ.copy()
        if not env.get("GH_TOKEN"):
            env["GH_TOKEN"] = token
        result = run(["gh", "pr", "view", branch], cwd=root, check=False, env=env)
        if result.returncode != 0:
            run(
                [
                    "gh",
                    "pr",
                    "create",
                    "--base",
                    args.base_branch,
                    "--head",
                    branch,
                    "--title",
                    f"autogen: {exp_id} {title}",
                    "--body-file",
                    str(pr_body_path),
                ],
                cwd=root,
                env=env,
            )
        for label in label_str.split(","):
            label = label.strip()
            if label:
                run(
                    ["gh", "pr", "edit", branch, "--add-label", label],
                    cwd=root,
                    check=False,
                    env=env,
                )
    elif args.publisher == "actions":
        patch_dir = root / "artifacts" / "patches"
        patch_dir.mkdir(parents=True, exist_ok=True)
        patch_path = patch_dir / f"{exp_id}_{slugify(title)}_{ts}.diff"
        meta_path = patch_dir / f"{exp_id}_{slugify(title)}_{ts}.meta.json"
        diff = run(
            ["git", "diff", f"origin/{args.base_branch}...HEAD"], cwd=root
        ).stdout
        patch_path.write_text(diff, encoding="utf-8")
        meta = {
            "title": f"autogen: {exp_id} {title}",
            "body": pr_body,
            "labels": [labels["autogen"], labels["auto_fix"]],
            "base_branch": args.base_branch,
        }
        meta_path.write_text(
            json.dumps(meta, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Patch written for publisher workflow: {patch_path}")
    else:
        print(
            "Publisher is set to 'none'. Commit created locally; no push or PR performed."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
