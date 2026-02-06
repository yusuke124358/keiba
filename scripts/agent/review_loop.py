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
AUTO_FIX_COMMENT_MARKER = "<!-- auto-fix: review_loop -->"


def state_root() -> Path:
    configured = os.environ.get("AUTO_FIX_STATE_DIR", "").strip()
    if configured:
        return Path(configured)
    if os.environ.get("GITHUB_ACTIONS") == "true":
        # Keep state outside the git workspace; actions/checkout may wipe untracked files.
        return Path.home() / ".clawdbot" / "auto_fix_state"
    return STATE_DIR


def run(cmd, cwd=None, check=True, capture_output=True, text=True, env=None):
    kwargs = {
        "cwd": cwd,
        "check": False,
        "capture_output": capture_output,
        "env": env,
    }
    if text:
        # gh/codex/git commonly emit UTF-8 on Windows; avoid cp932 decode failures.
        kwargs.update({"text": True, "encoding": "utf-8", "errors": "replace"})
    else:
        kwargs["text"] = False
    result = subprocess.run(cmd, **kwargs)
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
    try:
        return gh_json(["pr", "view", str(pr_number), "--json", ",".join(fields)])
    except RuntimeError as exc:
        message = str(exc)
        if 'Unknown JSON field: "reviewThreads"' in message:
            fallback_fields = [f for f in fields if f != "reviewThreads"]
            data = gh_json(
                ["pr", "view", str(pr_number), "--json", ",".join(fallback_fields)]
            )
            data["reviewThreads"] = []
            return data
        raise


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
        body = c.get("body", "") or ""
        # Avoid self-trigger loops: the workflow listens to issue_comment events,
        # and we also comment on the PR from this loop.
        if AUTO_FIX_COMMENT_MARKER in body:
            continue
        comments.append(
            {
                "id": c.get("databaseId") or 0,
                "author": (c.get("author") or {}).get("login", ""),
                "created_at": c.get("createdAt", ""),
                "body": body,
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
        name = c.get("name") or c.get("context") or ""
        if not name:
            continue
        checks.append(
            {
                "name": name,
                "status": c.get("status") or c.get("state") or "",
                "conclusion": c.get("conclusion") or c.get("state") or "",
                "completed_at": c.get("completedAt") or c.get("startedAt") or "",
                "details_url": c.get("detailsUrl") or c.get("targetUrl") or "",
                "summary": c.get("summary") or c.get("description") or "",
            }
        )
    return checks


RUN_ID_RE = re.compile(r"/actions/runs/(\d+)")


def attach_failed_check_logs(checks, max_checks=3, max_chars=12000):
    attached = 0
    for c in checks:
        if attached >= max_checks:
            return
        conclusion = str(c.get("conclusion") or "").upper()
        if conclusion != "FAILURE":
            continue
        url = str(c.get("details_url") or "")
        match = RUN_ID_RE.search(url)
        if not match:
            continue
        run_id = match.group(1)
        result = run(["gh", "run", "view", run_id, "--log-failed"], check=False)
        text = (result.stdout or "") + (("\n" + result.stderr) if result.stderr else "")
        text = text.strip()
        if max_chars and len(text) > max_chars:
            text = text[-max_chars:]
        c["run_id"] = run_id
        c["log_failed_excerpt"] = text
        attached += 1


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

    bundle_text = ""
    if bundle_path.exists():
        bundle_text = bundle_path.read_text(encoding="utf-8").strip()
    reviewer_prompt = "\n".join(
        [
            reviewer_base,
            "",
            f"Input bundle JSON: {bundle_path}",
            "Input bundle JSON content:",
            bundle_text or "{}",
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


def codex_help(codex_bin: str) -> str:
    result = subprocess.run(
        [codex_bin, "exec", "--help"], capture_output=True, text=True
    )
    if result.returncode != 0:
        return ""
    return (result.stdout or "") + (result.stderr or "")


def flag_supported(help_text: str, flag: str) -> bool:
    return flag in help_text


def run_codex(prompt_text, schema_path, output_path, log_path, codex_bin):
    help_text = codex_help(codex_bin)
    cmd = [codex_bin, "exec"]
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
            # Pass the prompt via stdin to avoid Windows argument parsing/encoding issues.
            result = subprocess.run(
                cmd + ["-"],
                input=prompt_text,
                stdout=log,
                stderr=log,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=int(os.environ.get("CODEX_TIMEOUT_SECONDS", "1800")),
            )
    else:
        result = subprocess.run(
            cmd + ["-"],
            input=prompt_text,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=int(os.environ.get("CODEX_TIMEOUT_SECONDS", "1800")),
        )
        with open(log_path, "w", encoding="utf-8") as log:
            log.write(result.stdout or "")
            if result.stderr:
                log.write("\n")
                log.write(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"codex exec failed with code {result.returncode}. See {log_path}"
        )

    if not supports_output_last:
        output_path.write_text(result.stdout or "", encoding="utf-8")


def ensure_branch(root, head_ref):
    # Use an explicit refspec so this works even if the checkout action configures
    # a narrow fetch spec (e.g., only default branch).
    run(
        [
            "git",
            "fetch",
            "origin",
            f"+refs/heads/{head_ref}:refs/remotes/origin/{head_ref}",
        ],
        cwd=root,
    )
    run(["git", "checkout", "-B", head_ref, f"origin/{head_ref}"], cwd=root)


def git_status_clean(root):
    return run(["git", "status", "--porcelain"], cwd=root).stdout.strip() == ""


def python_exe(root: Path) -> str:
    if os.name == "nt":
        venv_py = root / "py64_analysis" / ".venv" / "Scripts" / "python.exe"
    else:
        venv_py = root / "py64_analysis" / ".venv" / "bin" / "python"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable or "python"


RUFF_WOULD_REFORMAT_RE = re.compile(r"Would reformat:\s+([^\s]+)")
RUFF_CHECK_PATH_RE = re.compile(r"^([^\s:]+\.py):\d+:\d+:", re.MULTILINE)


def extract_ruff_reformat_paths(bundle: dict) -> list[str]:
    signals = bundle.get("signals") or {}
    checks = signals.get("checks") or []
    paths: set[str] = set()
    for check in checks:
        excerpt = str(check.get("log_failed_excerpt") or "")
        for match in RUFF_WOULD_REFORMAT_RE.finditer(excerpt):
            path = match.group(1).strip()
            if path:
                paths.add(path)
    return sorted(paths)


def extract_ruff_check_paths(bundle: dict) -> list[str]:
    signals = bundle.get("signals") or {}
    checks = signals.get("checks") or []
    paths: set[str] = set()
    for check in checks:
        excerpt = str(check.get("log_failed_excerpt") or "")
        for match in RUFF_CHECK_PATH_RE.finditer(excerpt):
            path = match.group(1).strip()
            if path:
                paths.add(path)
    return sorted(paths)


def run_ruff_format(root: Path, paths: list[str]) -> None:
    if not paths:
        return
    py = python_exe(root)
    subprocess.run([py, "-m", "ruff", "format", *paths], cwd=root, check=True)


def run_ruff_check_fix(root: Path, paths: list[str]) -> int:
    """
    Apply safe auto-fixes for ruff check failures.

    ruff may still exit non-zero if remaining (non-fixable) issues exist; callers
    should treat this as best-effort and let CI determine final pass/fail.
    """
    if not paths:
        return 0
    py = python_exe(root)
    result = subprocess.run(
        [py, "-m", "ruff", "check", "--fix", "--quiet", *paths],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return int(result.returncode or 0)


def ruff_is_clean(root: Path, paths: list[str]) -> bool:
    if not paths:
        return True
    py = python_exe(root)
    fmt = subprocess.run(
        [py, "-m", "ruff", "format", "--check", *paths],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if fmt.returncode != 0:
        return False
    chk = subprocess.run(
        [py, "-m", "ruff", "check", "--quiet", *paths],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return chk.returncode == 0


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


def push_head_ref(root, head_ref, max_attempts=3):
    attempts = max(1, int(max_attempts))
    for attempt in range(1, attempts + 1):
        result = run(
            ["git", "push", "origin", f"HEAD:{head_ref}"],
            cwd=root,
            check=False,
        )
        if result.returncode == 0:
            return
        stderr = (result.stderr or "").strip()
        if attempt >= attempts:
            raise RuntimeError(
                f"git push failed ({result.returncode}): HEAD:{head_ref}\n{stderr}"
            )

        run(
            [
                "git",
                "fetch",
                "origin",
                f"+refs/heads/{head_ref}:refs/remotes/origin/{head_ref}",
            ],
            cwd=root,
            check=False,
        )
        rebase = run(
            ["git", "rebase", f"origin/{head_ref}"],
            cwd=root,
            check=False,
        )
        if rebase.returncode != 0:
            run(["git", "rebase", "--abort"], cwd=root, check=False)
            raise RuntimeError(
                "git push failed and rebase retry also failed.\n"
                f"push stderr:\n{stderr}\n"
                f"rebase stderr:\n{(rebase.stderr or '').strip()}"
            )


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
        AUTO_FIX_COMMENT_MARKER,
        "",
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
    state_root().mkdir(parents=True, exist_ok=True)

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

    state_path = state_root() / f"state_pr_{pr_number}.json"
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

    # When CI is still failing but no new signals exist, scheduled runs should
    # continue attempting fixes. Bundle failing checks (with log excerpts) so
    # we can apply deterministic fixes (e.g., ruff formatting) without requiring
    # new comments/reviews/check reruns.
    failing_checks = [
        c for c in checks if str(c.get("conclusion") or "").upper() == "FAILURE"
    ]
    checks_for_bundle = new_checks or failing_checks
    attach_failed_check_logs(checks_for_bundle)

    if (
        not (new_comments or new_reviews or new_thread_comments or checks_for_bundle)
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
            "checks": checks_for_bundle,
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
    state_root().mkdir(parents=True, exist_ok=True)

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

    state_path = state_root() / f"state_pr_{pr_number}.json"
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
    countable_issue_ids = []
    for issue in issues:
        source = str(issue.get("source") or "")
        issue_type = str(issue.get("type") or "")
        # Do not treat mechanical CI gates (ruff formatting/lint) as "recurring
        # issues" that require human intervention. Those should be handled by
        # repeated auto-fix attempts, bounded by consecutive_fail/max_total.
        if source.startswith("checks:") and issue_type.startswith("ci/"):
            continue
        issue_id = issue.get("id")
        occurrences[issue_id] = int(occurrences.get(issue_id, 0)) + 1
        countable_issue_ids.append(issue_id)
    state["issue_occurrences"] = occurrences

    recurrence_limit = config["thresholds"]["recurrence"]
    if countable_issue_ids and any(
        occurrences.get(i, 0) >= recurrence_limit for i in countable_issue_ids
    ):
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
        fixer_report["status"] = "failed"
        fixer_report.setdefault("needs_human", False)
        if "failures" not in fixer_report:
            fixer_report["failures"] = []
        fixer_report["failures"].append("make ci failed")
        fixer_report["summary"] = "make ci failed"
        if state["consecutive_failures"] >= config["thresholds"]["consecutive_fail"]:
            fixer_report["needs_human"] = True
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
        push_head_ref(root, head_ref)

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

    run_dir = Path(run_meta["run_dir"])
    try:
        bundle = json.loads((run_dir / "input_bundle.json").read_text(encoding="utf-8"))
    except Exception:
        bundle = {}

    # Fast path: when CI logs clearly indicate a mechanical formatting failure,
    # fix it immediately without invoking the LLM stages. This keeps scheduled
    # auto-fix runs responsive and avoids execpolicy blocks on the runner.
    head_ref = ((bundle.get("pr") or {}).get("head_ref") or "").strip()
    ruff_paths = sorted(
        set(extract_ruff_reformat_paths(bundle)) | set(extract_ruff_check_paths(bundle))
    )
    if head_ref and ruff_paths:
        root = repo_root()
        ensure_branch(root, head_ref)
        run_ruff_check_fix(root, ruff_paths)
        run_ruff_format(root, ruff_paths)
        if not ruff_is_clean(root, ruff_paths):
            # Ruff still reports errors after safe auto-fixes/formatting. Continue
            # with the full LLM pipeline instead of returning early.
            pass
        else:
            review_items = {
                "issues": [
                    {
                        "id": "",
                        "source": "checks:verify",
                        "type": "ci/ruff",
                        "severity": "high",
                        "file": ruff_paths[0] if ruff_paths else None,
                        "line": None,
                        "message": "CI ruff gate failed (ruff format/check).",
                        "suggested_fix": "Run ruff check --fix and ruff format, then commit.",
                        "acceptance_check": "make ci passes (ruff format/check are clean).",
                    }
                ]
            }
            write_json(run_dir / "review_items.json", review_items)

            manager_decision = {
                "approved": True,
                "summary": "Apply ruff fixes from failing CI logs.",
                "tasks": [
                    {
                        "issue_id": "",
                        "decision": "DO",
                        "priority": "P0",
                        "risk": "low",
                        "rationale": "Ruff failure blocks verify/CI; safe mechanical change.",
                        "required_checks": ["checks:verify"],
                    }
                ],
                "automerge_eligible": True,
            }
            write_json(run_dir / "manager_decision.json", manager_decision)

            fixer_report = {
                "status": "success",
                "summary": "Applied deterministic CI fixes.",
                "actions": [
                    "ruff check --fix: " + ", ".join(ruff_paths),
                    "ruff format: " + ", ".join(ruff_paths),
                ],
                "tests": [],
                "artifacts": [],
                "failures": [],
                "needs_human": False,
                "next_steps": [],
            }
            write_json(run_dir / "fixer_report.json", fixer_report)
            return finalize(args, config)

    codex_bin = find_codex_bin()
    if not codex_bin:
        raise RuntimeError("codex CLI not found in PATH")
    ensure_codex_ready(codex_bin)

    reviewer_prompt = (run_dir / "reviewer_prompt.txt").read_text(encoding="utf-8")

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

    review_items_path = run_dir / "review_items.json"
    review_items_text = (
        review_items_path.read_text(encoding="utf-8").strip()
        if review_items_path.exists()
        else "{}"
    )
    # Fallback: if Reviewer returns no issues but CI checks are failing, synthesize
    # a minimal issue so Manager/Fixer stages can still make progress.
    signals = bundle.get("signals") or {}
    all_checks = signals.get("checks") or []
    failing_checks = [
        c for c in all_checks if str(c.get("conclusion") or "").upper() == "FAILURE"
    ]
    if failing_checks:
        try:
            review_items = json.loads(review_items_text or "{}")
        except Exception:
            review_items = {"issues": []}
        if not (review_items.get("issues") or []):
            ruff_paths = sorted(
                set(extract_ruff_reformat_paths(bundle))
                | set(extract_ruff_check_paths(bundle))
            )
            check_names = ", ".join(
                sorted({str(c.get("name") or "") for c in failing_checks if c.get("name")})
            )
            msg = f"CI is failing: {check_names}" if check_names else "CI is failing."
            suggested = "Inspect failing check logs and fix until `make ci` passes."
            issue_type = "ci/failure"
            if ruff_paths:
                msg = "CI is failing due to ruff format/check."
                suggested = (
                    "Fix remaining ruff check errors (e.g., long lines, whitespace) "
                    "and ensure ruff format/check pass."
                )
                issue_type = "ci/ruff"
            review_items = {
                "issues": [
                    {
                        "id": "",
                        "source": "checks:ci",
                        "type": issue_type,
                        "severity": "high",
                        "file": ruff_paths[0] if ruff_paths else None,
                        "line": None,
                        "message": msg,
                        "suggested_fix": suggested,
                        "acceptance_check": "make ci passes",
                    }
                ]
            }
            write_json(review_items_path, review_items)
            review_items_text = review_items_path.read_text(encoding="utf-8").strip()
    manager_base = load_prompt(PROMPTS_DIR / "manager.md")
    manager_prompt = "\n".join(
        [
            manager_base,
            "",
            f"Review items JSON path: {review_items_path}",
            "Review items JSON content:",
            review_items_text or "{}",
            "Apply business rules from AGENTS.md.",
        ]
    )
    (run_dir / "manager_prompt.txt").write_text(manager_prompt, encoding="utf-8")
    run_codex(
        manager_prompt,
        manager_schema,
        run_dir / "manager_decision.json",
        run_dir / "manager_codex.log",
        codex_bin,
    )
    # If Manager returns no DO tasks while CI is failing, override with a minimal
    # DO task so the Fixer stage can attempt to unblock CI.
    manager_decision_path = run_dir / "manager_decision.json"
    if failing_checks and manager_decision_path.exists():
        try:
            manager_decision = json.loads(
                manager_decision_path.read_text(encoding="utf-8").strip() or "{}"
            )
        except Exception:
            manager_decision = {}
        tasks = manager_decision.get("tasks") or []
        if not any(t.get("decision") == "DO" for t in tasks):
            manager_decision = {
                "approved": True,
                "summary": "CI is failing; force fixer to attempt unblock.",
                "tasks": [
                    {
                        "issue_id": "",
                        "decision": "DO",
                        "priority": "P0",
                        "risk": "low",
                        "rationale": "Auto-fix eligible PR with failing CI; attempt mechanical fixes.",
                        "required_checks": ["checks:verify"],
                    }
                ],
                "automerge_eligible": False,
            }
            write_json(manager_decision_path, manager_decision)

    # Ensure the fixer runs on the PR branch (not main). The orchestrator will
    # commit/push later, but the fixer stage must apply edits to the correct ref.
    head_ref = ((bundle.get("pr") or {}).get("head_ref") or "").strip()
    if head_ref:
        ensure_branch(repo_root(), head_ref)

    # Fast path: fix common mechanical CI failures without running the fixer LLM.
    # This avoids Codex execpolicy blocks and reduces latency.
    root = repo_root()
    fast_actions = []
    ruff_paths = sorted(
        set(extract_ruff_reformat_paths(bundle)) | set(extract_ruff_check_paths(bundle))
    )
    if ruff_paths:
        run_ruff_check_fix(root, ruff_paths)
        run_ruff_format(root, ruff_paths)
        if ruff_is_clean(root, ruff_paths):
            fast_actions.append("ruff check --fix: " + ", ".join(ruff_paths))
            fast_actions.append("ruff format: " + ", ".join(ruff_paths))

    if fast_actions:
        # Ensure finalize() sees at least one DO task so it can commit/push.
        manager_decision_path = run_dir / "manager_decision.json"
        try:
            manager_decision = json.loads(
                manager_decision_path.read_text(encoding="utf-8").strip() or "{}"
            )
        except Exception:
            manager_decision = {}
        tasks = manager_decision.get("tasks") or []
        if not any(t.get("decision") == "DO" for t in tasks):
            manager_decision = {
                "approved": True,
                "summary": "Applied deterministic CI fixes.",
                "tasks": [
                    {
                        "issue_id": "",
                        "decision": "DO",
                        "priority": "P0",
                        "risk": "low",
                        "rationale": "Mechanical CI fix applied; proceed to validate and publish.",
                        "required_checks": ["checks:verify"],
                    }
                ],
                "automerge_eligible": True,
            }
            write_json(manager_decision_path, manager_decision)
        fixer_report = {
            "status": "success",
            "summary": "Applied deterministic CI fixes.",
            "actions": fast_actions,
            "tests": [],
            "artifacts": [],
            "failures": [],
            "needs_human": False,
            "next_steps": [],
        }
        write_json(run_dir / "fixer_report.json", fixer_report)
        return finalize(args, config)

    manager_decision_path = run_dir / "manager_decision.json"
    manager_text = (
        manager_decision_path.read_text(encoding="utf-8").strip()
        if manager_decision_path.exists()
        else "{}"
    )
    fixer_base = load_prompt(PROMPTS_DIR / "fixer.md")
    fixer_prompt = "\n".join(
        [
            fixer_base,
            "",
            f"Manager decisions JSON path: {manager_decision_path}",
            "Manager decisions JSON content:",
            manager_text or "{}",
            "",
            f"Review items JSON path: {review_items_path}",
            "Review items JSON content:",
            review_items_text or "{}",
            "Do not commit or push; orchestrator handles that.",
        ]
    )
    (run_dir / "fixer_prompt.txt").write_text(fixer_prompt, encoding="utf-8")
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
