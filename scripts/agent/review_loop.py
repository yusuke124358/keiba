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
        "PyYAML is required to run review_loop. Install with: pip install pyyaml",
        file=sys.stderr,
    )
    raise

from issue_id import build_issue_id


CONFIG_PATH = Path("config/auto_fix.yml")
PROMPTS_DIR = Path("prompts")
STATE_DIR = Path("artifacts/agent")
CURRENT_RUN = STATE_DIR / "current_run.json"


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


def load_config():
    data = {}
    if CONFIG_PATH.exists():
        data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
    labels = data.get("labels") or {}
    thresholds = data.get("thresholds") or {}
    return {
        "labels": {
            "auto_fix": labels.get("auto_fix", "auto-fix"),
            "needs_human": labels.get("needs_human", "needs-human"),
        },
        "thresholds": {
            "consecutive_fail": int(thresholds.get("consecutive_fail", 3)),
            "recurrence": int(thresholds.get("recurrence", 2)),
            "max_total": int(thresholds.get("max_total", 10)),
        },
    }


def load_state(path: Path):
    if not path.exists():
        return {
            "last_comment_id": 0,
            "last_review_id": 0,
            "last_thread_comment_id": 0,
            "last_check_completed_at": "",
            "attempts_total": 0,
            "consecutive_failures": 0,
            "issue_occurrences": {},
        }
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(state, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )


def parse_ts(value):
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def format_ts(value):
    if not value:
        return ""
    if isinstance(value, dt.datetime):
        return value.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    return str(value)


def gh_json(args):
    result = run(["gh"] + args, check=False)
    if result.returncode != 0:
        stderr = result.stderr.strip() if result.stderr else ""
        raise RuntimeError(f"gh failed: {' '.join(args)}\n{stderr}")
    return json.loads(result.stdout or "{}")


def gh_pr_view(pr_number):
    fields = [
        "number",
        "title",
        "url",
        "headRefName",
        "baseRefName",
        "isCrossRepository",
        "labels",
        "comments",
        "reviews",
        "reviewThreads",
        "statusCheckRollup",
    ]
    return gh_json(["pr", "view", str(pr_number), "--json", ",".join(fields)])


def list_target_prs(label, needs_human_label):
    data = gh_json(
        [
            "pr",
            "list",
            "--state",
            "open",
            "--label",
            label,
            "--json",
            "number,labels,headRefName,baseRefName",
        ]
    )
    prs = []
    for pr in data:
        labels = {label.get("name") for label in pr.get("labels", [])}
        if needs_human_label in labels:
            continue
        prs.append(pr)
    return prs


def select_pr(prs):
    if not prs:
        return None
    prs_sorted = sorted(prs, key=lambda p: p.get("number", 0))
    return prs_sorted[0]


def extract_comments(raw_comments):
    comments = []
    for c in raw_comments or []:
        comments.append(
            {
                "id": c.get("databaseId") or 0,
                "author": (c.get("author") or {}).get("login", ""),
                "created_at": c.get("createdAt", ""),
                "body": c.get("body", ""),
                "url": c.get("url", ""),
                "source": "issue_comment",
            }
        )
    return comments


def extract_reviews(raw_reviews):
    reviews = []
    for r in raw_reviews or []:
        reviews.append(
            {
                "id": r.get("databaseId") or 0,
                "author": (r.get("author") or {}).get("login", ""),
                "submitted_at": r.get("submittedAt", ""),
                "state": r.get("state", ""),
                "body": r.get("body", ""),
                "source": "review",
            }
        )
    return reviews


def extract_thread_comments(raw_threads):
    out = []
    for thread in raw_threads or []:
        path = thread.get("path", "")
        line = thread.get("line") or thread.get("originalLine")
        for c in thread.get("comments", []) or []:
            out.append(
                {
                    "id": c.get("databaseId") or 0,
                    "author": (c.get("author") or {}).get("login", ""),
                    "created_at": c.get("createdAt", ""),
                    "body": c.get("body", ""),
                    "path": path,
                    "line": line,
                    "source": "review_thread",
                }
            )
    return out


def extract_checks(raw_checks):
    checks = []
    for c in raw_checks or []:
        checks.append(
            {
                "name": c.get("name", ""),
                "status": c.get("status", ""),
                "conclusion": c.get("conclusion", ""),
                "completed_at": c.get("completedAt", ""),
                "details_url": c.get("detailsUrl", ""),
                "summary": c.get("summary", ""),
            }
        )
    return checks


def filter_new_by_id(items, last_id):
    return [i for i in items if int(i.get("id") or 0) > int(last_id or 0)]


def filter_new_checks(checks, last_completed_at):
    last_dt = parse_ts(last_completed_at)
    if not last_dt:
        return checks
    out = []
    for c in checks:
        ts = parse_ts(c.get("completed_at"))
        if not ts or ts > last_dt:
            out.append(c)
    return out


def load_prompt(path):
    if not path.exists():
        raise RuntimeError(f"Missing prompt: {path}")
    return path.read_text(encoding="utf-8").strip()


def build_prompts(run_dir, bundle_path):
    reviewer_base = load_prompt(PROMPTS_DIR / "reviewer.md")
    manager_base = load_prompt(PROMPTS_DIR / "manager.md")
    fixer_base = load_prompt(PROMPTS_DIR / "fixer.md")

    reviewer_prompt = "\n".join(
        [
            reviewer_base,
            "",
            f"Input bundle JSON: {bundle_path}",
        ]
    )
    manager_prompt = "\n".join(
        [
            manager_base,
            "",
            f"Review items JSON: {run_dir / 'review_items.json'}",
            "Apply business rules from AGENTS.md.",
        ]
    )
    fixer_prompt = "\n".join(
        [
            fixer_base,
            "",
            f"Manager decisions: {run_dir / 'manager_decision.json'}",
            f"Review items: {run_dir / 'review_items.json'}",
            "Do not commit or push; orchestrator handles that.",
        ]
    )
    return reviewer_prompt, manager_prompt, fixer_prompt


def write_json(path, payload):
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )


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


def run_codex(prompt_text, schema_path, output_path, log_path, codex_bin):
    cmd = [
        codex_bin,
        "exec",
        "--ask-for-approval",
        "never",
        "--sandbox",
        "workspace-write",
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(output_path),
    ]
    with open(log_path, "w", encoding="utf-8") as log:
        result = subprocess.run(cmd + [prompt_text], stdout=log, stderr=log, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"codex exec failed with code {result.returncode}. See {log_path}"
        )


def ensure_branch(root, head_ref):
    run(["git", "fetch", "origin", head_ref], cwd=root)
    run(["git", "checkout", "-B", head_ref, f"origin/{head_ref}"], cwd=root)


def git_status_clean(root):
    return run(["git", "status", "--porcelain"], cwd=root).stdout.strip() == ""


def run_make_ci(root, base_ref):
    env = os.environ.copy()
    env["VERIFY_BASE"] = base_ref
    result = subprocess.run(["make", "ci"], cwd=root, env=env)
    return result.returncode


def normalize_issue_ids(issues):
    out = []
    seen = set()
    for issue in issues:
        source = issue.get("source", "")
        file_path = issue.get("file", "") or ""
        line = issue.get("line", "") or ""
        message = issue.get("message", "")
        issue_id = build_issue_id(source, file_path, line, message)
        issue["id"] = issue_id
        if issue_id in seen:
            continue
        seen.add(issue_id)
        out.append(issue)
    return out


def summarize_manager(decision):
    summary = decision.get("summary", "")
    tasks = decision.get("tasks", [])
    counts = {"DO": 0, "DEFER": 0, "REJECT": 0, "NEEDS_HUMAN": 0}
    for t in tasks:
        counts[t.get("decision", "")] = counts.get(t.get("decision", ""), 0) + 1
    return summary, counts


def comment_pr(pr_number, body_path):
    run(["gh", "pr", "comment", str(pr_number), "--body-file", str(body_path)])


def add_label(pr_number, label):
    run(["gh", "pr", "edit", str(pr_number), "--add-label", label], check=False)


def build_comment(manager_decision, fixer_report):
    manager_json = json.dumps(manager_decision, ensure_ascii=True, indent=2)
    fixer_json = json.dumps(fixer_report, ensure_ascii=True, indent=2)
    manager_summary, counts = summarize_manager(manager_decision)
    fixer_summary = fixer_report.get("summary", "")
    counts_line = (
        f"Counts: DO={counts.get('DO', 0)} "
        f"DEFER={counts.get('DEFER', 0)} "
        f"REJECT={counts.get('REJECT', 0)} "
        f"NEEDS_HUMAN={counts.get('NEEDS_HUMAN', 0)}"
    )
    lines = [
        "### MANAGER_DECISION",
        f"Summary: {manager_summary}",
        counts_line,
        "<details><summary>manager_decision.json</summary>",
        "",
        "```json",
        manager_json,
        "```",
        "</details>",
        "",
        "### FIXER_REPORT",
        f"Summary: {fixer_summary}",
        "<details><summary>fixer_report.json</summary>",
        "",
        "```json",
        fixer_json,
        "```",
        "</details>",
        "",
    ]
    return "\n".join(lines)


def ensure_current_run(run_meta):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    write_json(CURRENT_RUN, run_meta)


def collect(args, config):
    root = repo_root()
    os.chdir(root)
    STATE_DIR.mkdir(parents=True, exist_ok=True)

    auto_label = config["labels"]["auto_fix"]
    needs_label = config["labels"]["needs_human"]

    if args.pr:
        pr_data = gh_pr_view(args.pr)
        if pr_data.get("isCrossRepository"):
            ensure_current_run(
                {"should_run": False, "reason": "fork_pr", "pr": args.pr}
            )
            return None
        labels = {label.get("name") for label in pr_data.get("labels", [])}
        if auto_label not in labels or needs_label in labels:
            ensure_current_run(
                {"should_run": False, "reason": "label_not_eligible", "pr": args.pr}
            )
            return None
        pr = {
            "number": pr_data.get("number"),
            "headRefName": pr_data.get("headRefName"),
            "baseRefName": pr_data.get("baseRefName"),
        }
    else:
        pr = select_pr(list_target_prs(auto_label, needs_label))
        if not pr:
            ensure_current_run({"should_run": False, "reason": "no_pr"})
            return None
        pr_data = gh_pr_view(pr.get("number"))
        if pr_data.get("isCrossRepository"):
            ensure_current_run(
                {"should_run": False, "reason": "fork_pr", "pr": pr.get("number")}
            )
            return None

    pr_number = pr_data.get("number")
    head_ref = pr_data.get("headRefName")
    base_ref = pr_data.get("baseRefName")

    state_path = STATE_DIR / f"state_pr_{pr_number}.json"
    state = load_state(state_path)

    comments = extract_comments(pr_data.get("comments"))
    reviews = extract_reviews(pr_data.get("reviews"))
    thread_comments = extract_thread_comments(pr_data.get("reviewThreads"))
    checks = extract_checks(pr_data.get("statusCheckRollup"))

    new_comments = filter_new_by_id(comments, state.get("last_comment_id"))
    new_reviews = filter_new_by_id(reviews, state.get("last_review_id"))
    new_thread_comments = filter_new_by_id(
        thread_comments, state.get("last_thread_comment_id")
    )
    new_checks = filter_new_checks(checks, state.get("last_check_completed_at"))

    if (
        not (new_comments or new_reviews or new_thread_comments or new_checks)
        and not args.force
    ):
        ensure_current_run(
            {"should_run": False, "reason": "no_new_signals", "pr": pr_number}
        )
        return None

    ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = STATE_DIR / f"pr_{pr_number}" / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    bundle = {
        "pr": {
            "number": pr_number,
            "url": pr_data.get("url", ""),
            "title": pr_data.get("title", ""),
            "head_ref": head_ref,
            "base_ref": base_ref,
            "labels": [label.get("name") for label in pr_data.get("labels", [])],
        },
        "signals": {
            "comments": new_comments,
            "reviews": new_reviews,
            "review_thread_comments": new_thread_comments,
            "checks": new_checks,
        },
    }
    bundle_path = run_dir / "input_bundle.json"
    write_json(bundle_path, bundle)

    reviewer_prompt, manager_prompt, fixer_prompt = build_prompts(run_dir, bundle_path)

    (run_dir / "reviewer_prompt.txt").write_text(reviewer_prompt, encoding="utf-8")
    (run_dir / "manager_prompt.txt").write_text(manager_prompt, encoding="utf-8")
    (run_dir / "fixer_prompt.txt").write_text(fixer_prompt, encoding="utf-8")

    run_meta = {
        "should_run": True,
        "pr_number": pr_number,
        "run_dir": str(run_dir),
        "reviewer_prompt": str(run_dir / "reviewer_prompt.txt"),
        "manager_prompt": str(run_dir / "manager_prompt.txt"),
        "fixer_prompt": str(run_dir / "fixer_prompt.txt"),
        "review_items": str(run_dir / "review_items.json"),
        "manager_decision": str(run_dir / "manager_decision.json"),
        "fixer_report": str(run_dir / "fixer_report.json"),
    }
    ensure_current_run(run_meta)
    return run_meta


def normalize_review_items(run_dir):
    review_items_path = run_dir / "review_items.json"
    if not review_items_path.exists():
        raise RuntimeError("review_items.json missing for normalization.")
    review_items = json.loads(review_items_path.read_text(encoding="utf-8"))
    issues = normalize_issue_ids(review_items.get("issues", []))
    review_items["issues"] = issues
    write_json(review_items_path, review_items)
    return review_items


def finalize(args, config):
    root = repo_root()
    os.chdir(root)

    run_dir = Path(args.run_dir) if args.run_dir else None
    if not run_dir:
        if not CURRENT_RUN.exists():
            print("No current run metadata found.", file=sys.stderr)
            return 2
        run_meta = json.loads(CURRENT_RUN.read_text(encoding="utf-8"))
        if not run_meta.get("should_run"):
            print("No pending run.", file=sys.stderr)
            return 0
        run_dir = Path(run_meta["run_dir"])

    review_items_path = run_dir / "review_items.json"
    manager_decision_path = run_dir / "manager_decision.json"
    fixer_report_path = run_dir / "fixer_report.json"

    if not review_items_path.exists() or not manager_decision_path.exists():
        print("Missing review or manager outputs.", file=sys.stderr)
        return 2

    review_items = normalize_review_items(run_dir)
    issues = review_items.get("issues", [])

    manager_decision = json.loads(manager_decision_path.read_text(encoding="utf-8"))
    tasks = manager_decision.get("tasks", [])

    pr_number = manager_decision.get("pr_number")
    bundle = json.loads((run_dir / "input_bundle.json").read_text(encoding="utf-8"))
    if not pr_number:
        pr_number = bundle["pr"]["number"]

    pr_data = gh_pr_view(pr_number)
    head_ref = pr_data.get("headRefName")
    base_ref = pr_data.get("baseRefName")

    state_path = STATE_DIR / f"state_pr_{pr_number}.json"
    state = load_state(state_path)

    signals = bundle.get("signals", {})
    comment_ids = [c.get("id", 0) for c in signals.get("comments", [])]
    review_ids = [r.get("id", 0) for r in signals.get("reviews", [])]
    thread_ids = [c.get("id", 0) for c in signals.get("review_thread_comments", [])]
    check_times = [parse_ts(c.get("completed_at")) for c in signals.get("checks", [])]

    def _max_with_current(current, values):
        vals = [int(current or 0)] + [int(v or 0) for v in values]
        return max(vals) if vals else int(current or 0)

    state["last_comment_id"] = _max_with_current(
        state.get("last_comment_id", 0), comment_ids
    )
    state["last_review_id"] = _max_with_current(
        state.get("last_review_id", 0), review_ids
    )
    state["last_thread_comment_id"] = _max_with_current(
        state.get("last_thread_comment_id", 0), thread_ids
    )
    if check_times:
        latest = max([c for c in check_times if c], default=None)
        if latest:
            state["last_check_completed_at"] = format_ts(latest)

    if any(t.get("decision") == "NEEDS_HUMAN" for t in tasks):
        add_label(pr_number, config["labels"]["needs_human"])
        fixer_report = {
            "status": "skipped",
            "summary": "Manager marked tasks as NEEDS_HUMAN.",
            "actions": [],
            "tests": [],
            "artifacts": [],
            "failures": [],
            "needs_human": True,
            "next_steps": ["Human decision required."],
        }
        write_json(fixer_report_path, fixer_report)
        comment_body = build_comment(manager_decision, fixer_report)
        comment_path = run_dir / "comment.md"
        comment_path.write_text(comment_body, encoding="utf-8")
        comment_pr(pr_number, comment_path)
        save_state(state_path, state)
        return 0

    occurrences = state.get("issue_occurrences", {})
    for issue in issues:
        issue_id = issue.get("id")
        occurrences[issue_id] = int(occurrences.get(issue_id, 0)) + 1
    state["issue_occurrences"] = occurrences

    recurrence_limit = config["thresholds"]["recurrence"]
    if any(count >= recurrence_limit for count in occurrences.values()):
        add_label(pr_number, config["labels"]["needs_human"])
        state["consecutive_failures"] = int(state.get("consecutive_failures", 0)) + 1
        fixer_report = {
            "status": "failed",
            "summary": "Issue recurrence threshold reached.",
            "actions": [],
            "tests": [],
            "artifacts": [],
            "failures": ["issue recurrence threshold reached"],
            "needs_human": True,
            "next_steps": ["Investigate root cause or add tests."],
        }
        write_json(fixer_report_path, fixer_report)
        comment_body = build_comment(manager_decision, fixer_report)
        comment_path = run_dir / "comment.md"
        comment_path.write_text(comment_body, encoding="utf-8")
        comment_pr(pr_number, comment_path)
        save_state(state_path, state)
        return 0

    do_tasks = [t for t in tasks if t.get("decision") == "DO"]
    if not do_tasks:
        fixer_report = {
            "status": "skipped",
            "summary": "No DO tasks from manager.",
            "actions": [],
            "tests": [],
            "artifacts": [],
            "failures": [],
            "needs_human": False,
            "next_steps": [],
        }
        write_json(fixer_report_path, fixer_report)
        comment_body = build_comment(manager_decision, fixer_report)
        comment_path = run_dir / "comment.md"
        comment_path.write_text(comment_body, encoding="utf-8")
        comment_pr(pr_number, comment_path)
        save_state(state_path, state)
        return 0

    ensure_branch(root, head_ref)

    if head_ref == "main":
        add_label(pr_number, config["labels"]["needs_human"])
        return 2

    if fixer_report_path.exists():
        fixer_report = json.loads(fixer_report_path.read_text(encoding="utf-8"))
    else:
        fixer_report = {
            "status": "failed",
            "summary": "Fixer output missing.",
            "actions": [],
            "tests": [],
            "artifacts": [],
            "failures": ["fixer_report.json missing"],
            "needs_human": True,
            "next_steps": ["Re-run fixer stage or inspect logs."],
        }
        write_json(fixer_report_path, fixer_report)

    state["attempts_total"] = int(state.get("attempts_total", 0)) + 1

    run(["git", "fetch", "origin", base_ref], cwd=root)
    exit_code = run_make_ci(root, f"origin/{base_ref}")
    if exit_code != 0:
        state["consecutive_failures"] = int(state.get("consecutive_failures", 0)) + 1
        fixer_report.setdefault("status", "failed")
        if "failures" not in fixer_report:
            fixer_report["failures"] = []
        fixer_report["failures"].append("make ci failed")
        fixer_report["summary"] = "make ci failed"
        write_json(fixer_report_path, fixer_report)
        comment_body = build_comment(manager_decision, fixer_report)
        comment_path = run_dir / "comment.md"
        comment_path.write_text(comment_body, encoding="utf-8")
        comment_pr(pr_number, comment_path)
        save_state(state_path, state)
        if state["consecutive_failures"] >= config["thresholds"]["consecutive_fail"]:
            add_label(pr_number, config["labels"]["needs_human"])
        return exit_code

    state["consecutive_failures"] = 0

    status = run(["git", "status", "--porcelain"], cwd=root).stdout.strip()
    if status:
        run(
            ["git", "config", "user.email", "auto-fix@users.noreply.github.com"],
            cwd=root,
        )
        run(["git", "config", "user.name", "auto-fix-bot"], cwd=root)
        run(["git", "add", "-A"], cwd=root)
        commit_msg = f"autofix: pr-{pr_number} [agent-fix]"
        run(["git", "commit", "-m", commit_msg], cwd=root)

        token = os.environ.get("AUTO_FIX_PUSH_TOKEN")
        if not token:
            raise RuntimeError("AUTO_FIX_PUSH_TOKEN is required for push.")

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
        run(["git", "push", "origin", f"HEAD:{head_ref}"], cwd=root)

    comment_body = build_comment(manager_decision, fixer_report)
    comment_path = run_dir / "comment.md"
    comment_path.write_text(comment_body, encoding="utf-8")
    comment_pr(pr_number, comment_path)

    if fixer_report.get("needs_human"):
        add_label(pr_number, config["labels"]["needs_human"])

    max_total = config["thresholds"]["max_total"]
    if state["attempts_total"] >= max_total:
        add_label(pr_number, config["labels"]["needs_human"])

    save_state(state_path, state)
    return 0


def run_all(args, config):
    run_meta = collect(args, config)
    if not run_meta:
        return 0

    codex_bin = find_codex_bin()
    if not codex_bin:
        raise RuntimeError("codex CLI not found in PATH")
    ensure_codex_ready(codex_bin)

    run_dir = Path(run_meta["run_dir"])
    reviewer_prompt = (run_dir / "reviewer_prompt.txt").read_text(encoding="utf-8")
    manager_prompt = (run_dir / "manager_prompt.txt").read_text(encoding="utf-8")
    fixer_prompt = (run_dir / "fixer_prompt.txt").read_text(encoding="utf-8")

    review_schema = Path("schemas/agent/review_items.schema.json")
    manager_schema = Path("schemas/agent/manager_decision.schema.json")
    fixer_schema = Path("schemas/agent/fixer_report.schema.json")

    run_codex(
        reviewer_prompt,
        review_schema,
        run_dir / "review_items.json",
        run_dir / "reviewer_codex.log",
        codex_bin,
    )
    normalize_review_items(run_dir)
    run_codex(
        manager_prompt,
        manager_schema,
        run_dir / "manager_decision.json",
        run_dir / "manager_codex.log",
        codex_bin,
    )
    run_codex(
        fixer_prompt,
        fixer_schema,
        run_dir / "fixer_report.json",
        run_dir / "fixer_codex.log",
        codex_bin,
    )
    return finalize(args, config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pr", type=int, default=0)
    parser.add_argument(
        "--stage", default="all", choices=["all", "collect", "normalize", "finalize"]
    )
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config()

    if args.stage == "collect":
        collect(args, config)
        return 0
    if args.stage == "normalize":
        if not args.run_dir:
            raise RuntimeError("--run-dir is required for normalize stage")
        normalize_review_items(Path(args.run_dir))
        return 0
    if args.stage == "finalize":
        return finalize(args, config)

    return run_all(args, config)


if __name__ == "__main__":
    sys.exit(main())
