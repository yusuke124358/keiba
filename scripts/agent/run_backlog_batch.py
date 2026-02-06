#!/usr/bin/env python3
import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

try:
    import yaml
except Exception as exc:
    raise RuntimeError(
        "PyYAML is required to run run_backlog_batch. Install with: pip install pyyaml"
    ) from exc


def run(
    cmd,
    cwd=None,
    check=True,
    capture_output=False,
    text=True,
    env=None,
    timeout=None,
):
    result = subprocess.run(
        cmd,
        cwd=cwd,
        check=False,
        capture_output=capture_output,
        text=text,
        env=env,
        timeout=timeout,
    )
    if check and result.returncode != 0:
        stderr = result.stderr.strip() if result.stderr else ""
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\n{stderr}"
        )
    return result


def repo_root() -> Path:
    out = run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True
    ).stdout.strip()
    return Path(out)


def ensure_clean(root: Path, auto_stash: bool = False) -> None:
    status = run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True
    ).stdout.strip()
    if status:
        if not auto_stash:
            raise RuntimeError("Working tree not clean. Commit or stash changes first.")
        stash_msg = (
            "auto-stash run_backlog_batch "
            f"{dt.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}Z"
        )
        run(["git", "stash", "push", "-u", "-m", stash_msg], cwd=root)
        status = run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True
        ).stdout.strip()
        if status:
            raise RuntimeError("Working tree still not clean after auto-stash.")


def git_config_get(root: Path, key: str) -> str:
    result = run(
        ["git", "config", "--get", key],
        cwd=root,
        check=False,
        capture_output=True,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def ensure_git_identity(root: Path) -> None:
    if not git_config_get(root, "user.name"):
        run(["git", "config", "user.name", "autogen-bot"], cwd=root)
    if not git_config_get(root, "user.email"):
        run(
            ["git", "config", "user.email", "autogen@users.noreply.github.com"],
            cwd=root,
        )


def current_branch(root: Path) -> str:
    return run(
        ["git", "branch", "--show-current"],
        cwd=root,
        capture_output=True,
    ).stdout.strip()


def git_checkout(root: Path, branch: str) -> None:
    run(["git", "checkout", branch], cwd=root)


def git_pull_ff(root: Path, remote: str, branch: str) -> None:
    run(["git", "fetch", remote], cwd=root)
    run(["git", "pull", "--ff-only", remote, branch], cwd=root)


def git_merge_no_edit(root: Path, branch: str) -> None:
    run(["git", "merge", "--no-edit", branch], cwd=root)


def load_backlog(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "items" not in data:
        raise RuntimeError("Invalid backlog format: expected top-level 'items'")
    return data


def save_backlog(path: Path, data: dict) -> None:
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=False), encoding="utf-8"
    )


def normalize_change_scope(value) -> str:
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    if value is None:
        return ""
    return str(value)


def parse_utc_timestamp(value: str | None) -> dt.datetime | None:
    if not value:
        return None
    try:
        return dt.datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        return None


def reset_item_to_todo(item: dict) -> None:
    item["status"] = "todo"
    item.pop("picked_at", None)


def requeue_stale_in_progress(items: list[dict], stale_hours: float) -> list[dict]:
    now = dt.datetime.utcnow()
    requeued = []
    for item in items:
        status = str(item.get("status", "")).lower()
        if status != "in_progress":
            continue
        picked_at = parse_utc_timestamp(item.get("picked_at"))
        if stale_hours <= 0:
            is_stale = True
        elif picked_at is None:
            is_stale = True
        else:
            age = now - picked_at
            is_stale = age >= dt.timedelta(hours=stale_hours)
        if is_stale:
            reset_item_to_todo(item)
            requeued.append(item)
    return requeued


def to_seed(item: dict) -> dict:
    return {
        "id": item["id"],
        "title": item.get("title", ""),
        "hypothesis": item.get("hypothesis", ""),
        "change_scope": normalize_change_scope(item.get("change_scope")),
        "acceptance_criteria": item.get("acceptance_criteria") or [],
        "metrics": item.get("metrics") or [],
        "risk_level": item.get("risk_level") or "medium",
        "max_diff_size": int(item.get("max_diff_size") or 200),
    }


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "experiment"


def build_branch_name(run_id: str, title: str) -> str:
    return f"agent/{run_id}-{slugify(title)}"


def pick_items(items: list[dict], count: int) -> list[dict]:
    picked = []
    for item in items:
        status = str(item.get("status", "todo")).lower()
        if status == "todo":
            picked.append(item)
            if len(picked) >= count:
                break
    return picked


def update_item_status(item: dict, status: str, branch: str | None = None) -> None:
    item["status"] = status
    if status == "in_progress" and not item.get("picked_at"):
        item["picked_at"] = dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    if branch:
        item["branch"] = branch


def python_exe(root: Path) -> str:
    if os.name == "nt":
        venv_py = root / "py64_analysis" / ".venv" / "Scripts" / "python.exe"
    else:
        venv_py = root / "py64_analysis" / ".venv" / "bin" / "python"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable or "python"


def push_branch(
    root: Path,
    branch: str,
    token: str | None,
    max_attempts: int = 3,
) -> None:
    if token:
        remote_url = run(
            ["git", "remote", "get-url", "origin"], cwd=root, capture_output=True
        ).stdout.strip()
        if remote_url.startswith("git@"):
            match = re.match(r"git@github.com:(.+?/.+?)\\.git", remote_url)
            repo = match.group(1) if match else ""
            https_url = f"https://github.com/{repo}.git" if repo else remote_url
        else:
            https_url = remote_url
        push_url = https_url.replace("https://", f"https://x-access-token:{token}@")
        run(["git", "remote", "set-url", "--push", "origin", push_url], cwd=root)

    # Avoid spurious failures when other automation merges to the same branch
    # between our pull and push. We retry by rebasing onto the latest remote tip.
    if current_branch(root) != branch:
        git_checkout(root, branch)

    attempts = max(1, int(max_attempts))
    for attempt in range(1, attempts + 1):
        result = run(
            ["git", "push", "origin", branch],
            cwd=root,
            check=False,
            capture_output=True,
        )
        if result.returncode == 0:
            return

        stderr = (result.stderr or "").strip()
        if attempt >= attempts:
            raise RuntimeError(
                f"Command failed ({result.returncode}): git push origin {branch}\n{stderr}"
            )

        run(["git", "fetch", "origin", branch], cwd=root, check=False)
        rebase = run(
            ["git", "rebase", f"origin/{branch}"],
            cwd=root,
            check=False,
            capture_output=True,
        )
        if rebase.returncode != 0:
            run(["git", "rebase", "--abort"], cwd=root, check=False)
            raise RuntimeError(
                "git push failed and rebase retry also failed.\n"
                f"push stderr:\n{stderr}\n"
                f"rebase stderr:\n{(rebase.stderr or '').strip()}"
            )

        # Small backoff to reduce collisions if multiple pushes happen close together.
        time.sleep(2)


def publish_branch(root: Path, base_branch: str, title: str, body_file: Path) -> None:
    run(
        [
            python_exe(root),
            "scripts/publish/publisher.py",
            "--base",
            base_branch,
            "--title",
            title,
            "--body-file",
            str(body_file),
        ],
        cwd=root,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backlog", default="experiments/backlog.yml")
    parser.add_argument("--base-branch", default="")
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--push-base", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument(
        "--stale-in-progress-hours",
        type=float,
        default=None,
        help="Requeue in-progress items older than this many hours (0 = all).",
    )
    parser.add_argument(
        "--item-timeout-minutes",
        type=float,
        default=None,
        help="Timeout per experiment run in minutes (0 disables).",
    )
    parser.add_argument(
        "--requeue-on-timeout",
        action="store_true",
        help="Put items back to todo when an experiment times out.",
    )
    parser.add_argument(
        "--auto-stash",
        action="store_true",
        help="Auto-stash local changes if the working tree is dirty.",
    )
    args = parser.parse_args()

    root = repo_root()
    os.chdir(root)

    ensure_clean(root, auto_stash=args.auto_stash)
    ensure_git_identity(root)
    base_branch = args.base_branch or current_branch(root)
    if not base_branch:
        raise RuntimeError("Unable to determine base branch.")
    git_checkout(root, base_branch)
    git_pull_ff(root, args.remote, base_branch)

    token = os.environ.get("AUTO_FIX_PUSH_TOKEN") or os.environ.get("GH_TOKEN")
    if args.push_base and not token:
        raise RuntimeError("AUTO_FIX_PUSH_TOKEN or GH_TOKEN required for --push-base.")

    backlog_path = root / args.backlog
    data = load_backlog(backlog_path)
    items = data.get("items", [])
    if args.stale_in_progress_hours is not None:
        requeued = requeue_stale_in_progress(items, args.stale_in_progress_hours)
        if requeued:
            print(f"Requeued {len(requeued)} stale in-progress items.")
            save_backlog(backlog_path, data)
            run(["git", "add", str(backlog_path)], cwd=root)
            run(
                [
                    "git",
                    "commit",
                    "-m",
                    f"chore: requeue {len(requeued)} stale backlog items",
                ],
                cwd=root,
            )
            if args.push_base:
                push_branch(root, base_branch, token)
    picked = pick_items(items, args.count)
    if not picked:
        print("No todo backlog items found.")
        return 0

    timeout_seconds = None
    if args.item_timeout_minutes is not None and args.item_timeout_minutes > 0:
        timeout_seconds = int(args.item_timeout_minutes * 60)

    for item in picked:
        exp_id = item.get("id")
        if not exp_id:
            raise RuntimeError("Backlog item missing id.")

        print(f"=== Running {exp_id} ===")
        try:
            ensure_clean(root, auto_stash=args.auto_stash)
            git_checkout(root, base_branch)
            git_pull_ff(root, args.remote, base_branch)

            seed = to_seed(item)
            seed_path = root / "artifacts" / "agent" / "seed_selected.yaml"
            seed_path.parent.mkdir(parents=True, exist_ok=True)
            seed_path.write_text(
                yaml.safe_dump([seed], sort_keys=False, allow_unicode=False),
                encoding="utf-8",
            )

            ts = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            plan_path = root / "artifacts" / "agent" / f"plan_{exp_id}_{ts}.json"
            run(
                [
                    python_exe(root),
                    "scripts/agent/plan_next_experiment.py",
                    "--seed",
                    str(seed_path),
                    "--out",
                    str(plan_path),
                ],
                cwd=root,
            )
            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            branch_name = build_branch_name(plan["run_id"], plan["title"])

            update_item_status(item, "in_progress", branch=branch_name)
            save_backlog(backlog_path, data)
            run(["git", "add", str(backlog_path)], cwd=root)
            run(["git", "commit", "-m", f"chore: mark {exp_id} in progress"], cwd=root)
            if args.push_base:
                push_branch(root, base_branch, token)

            run(
                [
                    python_exe(root),
                    "scripts/agent/run_experiment.py",
                    "--plan",
                    str(plan_path),
                ],
                cwd=root,
                timeout=timeout_seconds,
            )
            exp_branch = current_branch(root)

            git_checkout(root, base_branch)
            update_item_status(item, "done", branch=exp_branch)
            save_backlog(backlog_path, data)
            run(["git", "add", str(backlog_path)], cwd=root)
            run(["git", "commit", "-m", f"chore: mark {exp_id} done"], cwd=root)
            if args.push_base:
                push_branch(root, base_branch, token)

            if exp_branch:
                git_checkout(root, exp_branch)
                git_merge_no_edit(root, base_branch)
                if args.publish:
                    pr_body_path = (
                        root / "artifacts" / "agent" / f"pr_body_{exp_id}_{ts}.md"
                    )
                    pr_body_path.parent.mkdir(parents=True, exist_ok=True)
                    pr_body_path.write_text(
                        "\n".join(
                            [
                                f"# {exp_id} {plan['title']}",
                                "",
                                f"- Run ID: {plan['run_id']}",
                                f"- Experiment log: docs/experiments/{plan['run_id']}.md",
                                f"- Metrics: {plan['metrics_path']}",
                            ]
                        ),
                        encoding="utf-8",
                    )
                    publish_branch(
                        root,
                        base_branch,
                        f"autogen: {exp_id} {plan['title']}",
                        pr_body_path,
                    )
                git_checkout(root, base_branch)
        except subprocess.TimeoutExpired as exc:
            ensure_clean(root, auto_stash=args.auto_stash)
            git_checkout(root, base_branch)
            if args.requeue_on_timeout:
                reset_item_to_todo(item)
                status_label = "requeued"
            else:
                update_item_status(item, "failed")
                status_label = "failed"
            save_backlog(backlog_path, data)
            run(["git", "add", str(backlog_path)], cwd=root)
            run(
                [
                    "git",
                    "commit",
                    "-m",
                    f"chore: mark {exp_id} {status_label} after timeout",
                ],
                cwd=root,
            )
            if args.push_base:
                push_branch(root, base_branch, token)
            print(
                f"Timed out on {exp_id} after {timeout_seconds}s.",
                file=sys.stderr,
            )
            if not args.continue_on_failure:
                raise RuntimeError(f"Timeout on {exp_id}") from exc
        except Exception as exc:
            git_checkout(root, base_branch)
            # If we fail before marking the item in progress (e.g., planning failures),
            # keep the item as todo so reruns don't require manual backlog edits.
            if str(item.get("status", "")).lower() == "todo":
                print(
                    f"Failed on {exp_id} before marking in progress: {exc}",
                    file=sys.stderr,
                )
                if not args.continue_on_failure:
                    raise
                continue
            update_item_status(item, "failed")
            save_backlog(backlog_path, data)
            run(["git", "add", str(backlog_path)], cwd=root)
            run(["git", "commit", "-m", f"chore: mark {exp_id} failed"], cwd=root)
            if args.push_base:
                push_branch(root, base_branch, token)
            print(f"Failed on {exp_id}: {exc}", file=sys.stderr)
            if not args.continue_on_failure:
                raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
