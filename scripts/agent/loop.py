#!/usr/bin/env python3
import argparse
import datetime as dt
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
        "PyYAML is required to run the agent loop. Install with: pip install pyyaml",
        file=sys.stderr,
    )
    raise


def run(cmd, cwd=None, check=True, capture_output=True, text=True):
    result = subprocess.run(
        cmd, cwd=cwd, check=False, capture_output=capture_output, text=text
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
            "Working tree not clean. Commit or stash changes before running the loop."
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


def run_codex(
    root, prompt_text, schema_path, output_last_message, profile, log_path, codex_bin
):
    cmd = [
        codex_bin,
        "exec",
        "--profile",
        profile,
        "--full-auto",
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(output_last_message),
    ]

    with open(log_path, "w", encoding="utf-8") as log:
        result = subprocess.run(
            cmd + [prompt_text],
            cwd=root,
            stdout=log,
            stderr=log,
            text=True,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"codex exec failed with code {result.returncode}. See {log_path}"
        )


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
        cmd = [
            "powershell",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(root / "scripts" / "verify.ps1"),
        ]
    else:
        cmd = ["bash", str(root / "scripts" / "verify.sh")]
    result = subprocess.run(cmd, cwd=root)
    if result.returncode != 0:
        raise RuntimeError(f"verify failed with code {result.returncode}")


def write_patch(root, base_ref, patch_path):
    diff = run(["git", "diff", f"{base_ref}...HEAD"], cwd=root).stdout
    patch_path.write_text(diff, encoding="utf-8")
    return patch_path


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
    parser.add_argument(
        "--publisher", choices=["none", "local", "actions"], default="none"
    )
    parser.add_argument("--patch-dir", default="artifacts/patches")
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
        run_codex(
            root,
            prompt_text,
            root / args.schema,
            codex_last,
            args.profile,
            codex_log,
            codex_bin,
        )

    total_diff = diff_size(root)
    if total_diff > max_diff_size:
        run(["git", "reset"], cwd=root, check=False)
        raise RuntimeError(
            f"Diff size {total_diff} exceeds max_diff_size {max_diff_size}"
        )

    if not args.skip_tests:
        run_verify(root)

    run(["git", "add", "-A"], cwd=root)
    status = run(["git", "status", "--porcelain"], cwd=root).stdout.strip()
    if not status:
        print("No changes to commit.")
        return 0

    commit_msg = f"agent: {exp_id} {slugify(title)}"
    run(["git", "commit", "-m", commit_msg], cwd=root)

    if args.publisher == "actions":
        patch_dir = Path(args.patch_dir)
        patch_dir.mkdir(parents=True, exist_ok=True)
        patch_path = patch_dir / f"{exp_id}_{slugify(title)}_{ts}.diff"
        write_patch(root, args.base_branch, patch_path)
        print(f"Patch written for publisher workflow: {patch_path}")
        return 0

    if args.publisher == "local":
        if os.name == "nt":
            publisher = root / "scripts" / "publish" / "local_push.ps1"
            run(
                ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(publisher)],
                cwd=root,
            )
        else:
            publisher = root / "scripts" / "publish" / "local_push.sh"
            run(["bash", str(publisher)], cwd=root)
        return 0

    print(
        "Publisher is set to 'none'. Commit created locally; no push or PR performed."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
