#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run(cmd, cwd=None, check=True):
    result = subprocess.run(cmd, cwd=cwd, check=False)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")
    return result


def find_codex_bin() -> str:
    if os.name == "nt":
        for name in ("codex.cmd", "codex.exe", "codex"):
            path = shutil.which(name)
            if path:
                return path
    path = shutil.which("codex")
    if not path:
        raise RuntimeError("codex CLI not found in PATH")
    return path


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "experiment"


def ensure_clean(root: Path) -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    if status:
        raise RuntimeError("Working tree not clean. Commit or stash changes first.")


def render_experiment_log(template: Path, out_path: Path, plan: dict, result: dict) -> None:
    text = template.read_text(encoding="utf-8")
    text = text.replace("<id>", plan["run_id"])
    text = text.replace("<title>", plan["title"])
    text = text.replace("<write the hypothesis>", plan["hypothesis"])
    text = text.replace("<low|medium|high>", plan["risk_level"])
    text = text.replace("<int>", str(plan["max_diff_size"]))
    metrics = result["metrics"]
    text = text.replace("<value>", str(metrics["roi"]), 1)
    text = text.replace("<value>", str(metrics["total_stake"]), 1)
    text = text.replace("<value>", str(metrics["n_bets"]), 1)
    text = text.replace("YYYY-MM-DD to YYYY-MM-DD", metrics["test_period"])
    text = text.replace("<value>", str(metrics["max_drawdown"]), 1)
    text = text.replace("yes|no", metrics["rolling"])
    text = text.replace("<if rolling, else N/A>", metrics["design_window"], 1)
    text = text.replace("<if rolling, else N/A>", metrics["eval_window"], 1)
    text = text.replace("<if rolling, else N/A>", metrics["paired_delta"], 1)
    text = text.replace("yes|no", metrics["pooled_vs_step14_mismatch"])
    text = text.replace("step14|pooled", metrics["preferred_roi"])
    text = text.replace("<path>", result["artifacts"]["metrics_json"], 1)
    text = text.replace("<path>", result["artifacts"]["comparison_json"], 1)
    text = text.replace("<path>", result["artifacts"]["report"], 1)
    out_path.write_text(text, encoding="utf-8")


def run_codex(root: Path, prompt_path: Path, plan: dict, profile: str, log_path: Path) -> None:
    codex_bin = find_codex_bin()
    prompt_text = prompt_path.read_text(encoding="utf-8")
    payload = json.dumps(plan, ensure_ascii=True)
    cmd = [codex_bin, "exec", "--profile", profile, "--full-auto", prompt_text + "\n\nPLAN_JSON:\n" + payload]
    with open(log_path, "w", encoding="utf-8") as log:
        result = subprocess.run(cmd, cwd=root, stdout=log, stderr=log, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"codex exec failed with code {result.returncode}. See {log_path}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True, help="Path to experiment_plan.json")
    p.add_argument("--profile", default="agent_loop")
    p.add_argument("--prompt", default="prompts/agent/fixer_implement.md")
    p.add_argument("--metrics-schema", default="schemas/agent/experiment_result.schema.json")
    args = p.parse_args()

    root = repo_root()
    ensure_clean(root)

    plan_path = Path(args.plan)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("decision") != "do":
        print("Plan decision is not 'do'; skipping.")
        return 0

    branch = f"agent/{plan['run_id']}-{slugify(plan['title'])}"
    run(["git", "checkout", "-b", branch], cwd=root)

    log_dir = root / "artifacts" / "agent"
    log_dir.mkdir(parents=True, exist_ok=True)
    codex_log = log_dir / f"{plan['run_id']}_implement.log"
    run_codex(root, root / args.prompt, plan, args.profile, codex_log)

    eval_cmd = plan["eval_command"]
    if not eval_cmd:
        raise RuntimeError("eval_command is empty.")
    run(eval_cmd, cwd=root, check=True)

    metrics_path = root / plan["metrics_path"]
    if not metrics_path.exists():
        raise RuntimeError(f"metrics_path not found: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    result = {
        "run_id": plan["run_id"],
        "seed_id": plan["seed_id"],
        "title": plan["title"],
        "status": metrics.get("status", "inconclusive"),
        "metrics": metrics["metrics"],
        "artifacts": metrics["artifacts"],
    }
    result_path = root / "experiments" / "runs" / f"{plan['run_id']}.json"
    result_path.write_text(json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8")

    template = root / "docs" / "experiments" / "_template.md"
    log_path = root / "docs" / "experiments" / f"{plan['run_id']}.md"
    render_experiment_log(template, log_path, plan, result)

    run(["git", "add", "-A"], cwd=root)
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    if not status:
        print("No changes to commit.")
        return 0

    commit_msg = f"agent: {plan['run_id']} {slugify(plan['title'])}"
    run(["git", "commit", "-m", commit_msg], cwd=root)
    print(f"Committed {plan['run_id']} on {branch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
