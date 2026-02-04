#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import subprocess
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run(cmd, cwd=None, check=True):
    if isinstance(cmd, list):
        if cmd and len(cmd) == 1 and isinstance(cmd[0], str):
            result = subprocess.run(cmd[0], cwd=cwd, check=False, shell=True)
            if check and result.returncode != 0:
                raise RuntimeError(f"Command failed ({result.returncode}): {cmd[0]}")
            return result
        result = subprocess.run(cmd, cwd=cwd, check=False)
        if check and result.returncode != 0:
            raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(cmd)}")
        return result
    if isinstance(cmd, str):
        result = subprocess.run(cmd, cwd=cwd, check=False, shell=True)
    else:
        result = subprocess.run(cmd, cwd=cwd, check=False)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {cmd}")
    return result


def run_shell_commands(commands, cwd=None):
    index = 0
    while index < len(commands):
        cmd = commands[index]
        next_cmd = commands[index + 1] if index + 1 < len(commands) else None
        ps_env = isinstance(cmd, str) and re.match(r"^\\s*\\$env:[A-Za-z_][A-Za-z0-9_]*\\s*=", cmd)
        if ps_env and next_cmd:
            ps_line = f"{cmd}; {next_cmd}".replace('"', '`"')
            cmd = f"powershell -ExecutionPolicy Bypass -Command \"{ps_line}\""
            index += 2
        else:
            index += 1
        if isinstance(cmd, str) and "py64_analysis\\.venv\\Scripts\\python.exe" in cmd:
            if cwd is None or not (Path(cwd) / "py64_analysis" / ".venv" / "Scripts" / "python.exe").exists():
                cmd = cmd.replace("py64_analysis\\.venv\\Scripts\\python.exe", "python")
        if isinstance(cmd, str):
            lowered = cmd.strip().lower()
            if lowered in {"ci", "make ci"} and os.name == "nt":
                cmd = "powershell -ExecutionPolicy Bypass -File scripts/ci.ps1"
        if isinstance(cmd, str):
            stripped = cmd.strip()
            if not stripped:
                continue
            if re.match(r"^(?:\\.\\\\)?py64_analysis[\\\\/].*\\.py(\\s|$)", stripped):
                cmd = f"python {stripped}"
            elif stripped.lower().startswith(("py64_analysis/", "py64_analysis\\")):
                cmd = f"python {stripped}"
        result = subprocess.run(cmd, cwd=cwd, check=False, shell=True)
        if result.returncode != 0:
            raise RuntimeError(f"Command failed ({result.returncode}): {cmd}")


def substitute_placeholders(text: str, run_id: str) -> str:
    if not text:
        return text
    for token in ("<run_id>", "{run_id}"):
        text = text.replace(token, run_id)
    for token in ("<CONFIG_PATH>", "<config_path>"):
        text = text.replace(token, "")
    text = re.sub(r"--config\\s+<CONFIG_PATH>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"--config\\s+<config_path>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"--config=\\s*<CONFIG_PATH>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"--config=\\s*<config_path>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\\s+", " ", text).strip()
    return text


def substitute_eval_command(text: str, run_id: str) -> str:
    if not text:
        return text
    run_dir = f"data/holdout_runs/{run_id}"
    for token in ("<RUN_DIR>", "<run_dir>", "{run_dir}"):
        text = text.replace(token, run_dir)
    return substitute_placeholders(text, run_id)


def substitute_metrics_path(text: str, run_id: str) -> str:
    if not text:
        return text
    for token in ("<RUN_DIR>", "<run_dir>", "{run_dir}"):
        text = text.replace(token, run_id)
    text = substitute_placeholders(text, run_id)
    text = text.replace("data/holdout_runs/data/holdout_runs/", "data/holdout_runs/")
    return text


def default_holdout_command(run_id: str) -> str:
    return (
        "python py64_analysis/scripts/run_holdout.py"
        " --train-start 2020-01-01 --train-end 2022-12-31"
        " --valid-start 2023-01-01 --valid-end 2023-12-31"
        " --test-start 2024-01-01 --test-end 2024-12-31"
        f" --name {run_id} --out-dir data/holdout_runs/{run_id}"
    )


def normalize_eval_command(cmd: str, run_id: str) -> str:
    cmd = substitute_eval_command(cmd, run_id)
    if "compare_metrics_json.py" in cmd and (
        "<scenario>" in cmd or "baselines/<scenario>" in cmd
    ):
        return ""
    if "run_holdout.py" in cmd and "--train-start" not in cmd:
        return default_holdout_command(run_id)
    return cmd


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


def _format_metric(value):
    return "N/A" if value is None else value


def normalize_metrics(metrics_json: dict) -> dict:
    backtest = metrics_json.get("backtest")
    if backtest is None:
        raise RuntimeError("metrics.json missing required 'backtest' section.")
    run_kind = metrics_json.get("run_kind", "holdout")
    split = metrics_json.get("split", {})
    test_split = split.get("test", {}) if isinstance(split, dict) else {}
    test_period = f"{test_split.get('start', 'N/A')} to {test_split.get('end', 'N/A')}"

    roi = backtest.get("roi")
    step14 = metrics_json.get("step14") or {}
    step14_roi = step14.get("roi")
    pooled_vs_step14_mismatch = "no"
    preferred_roi = "pooled"
    if run_kind == "rolling_holdout" and step14_roi is not None and roi is not None:
        if (roi >= 0) != (step14_roi >= 0):
            pooled_vs_step14_mismatch = "yes"
            preferred_roi = "step14"
        else:
            preferred_roi = "step14"

    return {
        "roi": _format_metric(roi),
        "total_stake": _format_metric(backtest.get("total_stake")),
        "n_bets": _format_metric(backtest.get("n_bets")),
        "max_drawdown": _format_metric(backtest.get("max_drawdown")),
        "test_period": test_period,
        "rolling": "yes" if run_kind == "rolling_holdout" else "no",
        "design_window": "N/A",
        "eval_window": "N/A",
        "paired_delta": "N/A",
        "pooled_vs_step14_mismatch": pooled_vs_step14_mismatch,
        "preferred_roi": preferred_roi,
    }


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


def render_experiment_log(
    template: Path, out_path: Path, plan: dict, result: dict
) -> None:
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


def run_codex(
    root: Path, prompt_path: Path, plan: dict, profile: str, log_path: Path
) -> None:
    codex_bin = find_codex_bin()
    prompt_text = prompt_path.read_text(encoding="utf-8")
    payload = json.dumps(plan, ensure_ascii=True)
    cmd = [
        codex_bin,
        "exec",
        "--profile",
        profile,
        "--full-auto",
        prompt_text + "\n\nPLAN_JSON:\n" + payload,
    ]
    with open(log_path, "w", encoding="utf-8") as log:
        result = subprocess.run(cmd, cwd=root, stdout=log, stderr=log, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"codex exec failed with code {result.returncode}. See {log_path}"
        )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True, help="Path to experiment_plan.json")
    p.add_argument("--profile", default="agent_loop")
    p.add_argument("--prompt", default="prompts/agent/fixer_implement.md")
    p.add_argument(
        "--metrics-schema", default="schemas/agent/experiment_result.schema.json"
    )
    args = p.parse_args()

    root = repo_root()
    ensure_clean(root)

    plan_path = Path(args.plan)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("decision") != "do":
        print("Plan decision is not 'do'; skipping.")
        return 0
    run_id = plan.get("run_id", "")
    if not run_id:
        raise RuntimeError("plan.run_id is required.")

    branch = f"agent/{run_id}-{slugify(plan['title'])}"
    run(["git", "checkout", "-b", branch], cwd=root)

    log_dir = root / "artifacts" / "agent"
    log_dir.mkdir(parents=True, exist_ok=True)
    codex_log = log_dir / f"{plan['run_id']}_implement.log"
    run_codex(root, root / args.prompt, plan, args.profile, codex_log)

    eval_cmd = plan["eval_command"]
    if not eval_cmd:
        raise RuntimeError("eval_command is empty.")
    if isinstance(eval_cmd, list):
        eval_cmd = [normalize_eval_command(cmd, run_id) for cmd in eval_cmd]
        run_shell_commands(eval_cmd, cwd=root)
    else:
        eval_cmd = normalize_eval_command(eval_cmd, run_id)
        run(eval_cmd, cwd=root, check=True)

    metrics_path = root / substitute_metrics_path(plan["metrics_path"], run_id)
    if not metrics_path.exists():
        raise RuntimeError(f"metrics_path not found: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    normalized = normalize_metrics(metrics)
    run_dir = metrics.get("run_dir", f"data/holdout_runs/{run_id}")
    metrics_json_path = str(metrics_path.relative_to(root)).replace("\\", "/")
    report_path = "N/A"
    comparison_path = "N/A"
    report_candidate = root / run_dir / "report" / "backtest.md"
    if report_candidate.exists():
        report_path = str(report_candidate.relative_to(root)).replace("\\", "/")
    comparison_candidate = root / run_dir / "comparison.json"
    if comparison_candidate.exists():
        comparison_path = str(comparison_candidate.relative_to(root)).replace("\\", "/")

    result = {
        "run_id": run_id,
        "seed_id": plan["seed_id"],
        "title": plan["title"],
        "status": metrics.get("status", "inconclusive"),
        "metrics": normalized,
        "artifacts": {
            "metrics_json": metrics_json_path,
            "comparison_json": comparison_path,
            "report": report_path,
        },
    }
    result_path = root / "experiments" / "runs" / f"{plan['run_id']}.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8"
    )

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
