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
        ps_env = isinstance(cmd, str) and re.match(
            r"^\s*\$env:[A-Za-z_][A-Za-z0-9_]*\s*=", cmd
        )
        if ps_env and next_cmd:
            ps_line = f"{cmd}; {next_cmd}".replace('"', '`"')
            cmd = f'powershell -ExecutionPolicy Bypass -Command "{ps_line}"'
            index += 2
        else:
            index += 1
        if isinstance(cmd, str) and "py64_analysis\\.venv\\Scripts\\python.exe" in cmd:
            if (
                cwd is None
                or not (
                    Path(cwd) / "py64_analysis" / ".venv" / "Scripts" / "python.exe"
                ).exists()
            ):
                cmd = cmd.replace("py64_analysis\\.venv\\Scripts\\python.exe", "python")
        if isinstance(cmd, str):
            lowered = cmd.strip().lower()
            if lowered in {"ci", "make ci"} and os.name == "nt":
                cmd = "powershell -ExecutionPolicy Bypass -File scripts/ci.ps1"
        if isinstance(cmd, str):
            stripped = cmd.strip()
            if not stripped:
                continue
            lowered = stripped.lower()
            if lowered.startswith(
                (
                    "py64_analysis\\.venv\\scripts\\python.exe",
                    "py64_analysis/.venv/scripts/python.exe",
                )
            ):
                pass
            elif re.match(r"^(?:\\.\\\\)?py64_analysis[\\\\/].*\\.py(\\s|$)", stripped):
                cmd = f"python {stripped}"
            elif lowered.startswith(("py64_analysis/", "py64_analysis\\")):
                cmd = f"python {stripped}"
        result = subprocess.run(cmd, cwd=cwd, check=False, shell=True)
        if result.returncode != 0:
            is_compare = (
                isinstance(cmd, str)
                and "compare_metrics_json.py" in cmd.replace("\\", "/").lower()
            )
            if is_compare and result.returncode in {1, 2}:
                # `compare_metrics_json.py` uses exit code 1/2 to encode gate outcomes.
                # We still want to persist results and decision metadata for reject/needs-human.
                print(
                    f"[warn] compare_metrics_json exited with code {result.returncode}; continuing."
                )
                continue
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
    text = text.replace("path\\to\\run_dir", f"data/holdout_runs/{run_id}")
    text = text.replace("path/to/run_dir", f"data/holdout_runs/{run_id}")
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
    if "path\\to\\config.yaml" in cmd or "path/to/config.yaml" in cmd:
        return ""
    if "path\\to\\run_dir" in cmd or "path/to/run_dir" in cmd:
        if "run_holdout.py" in cmd:
            return default_holdout_command(run_id)
    if "compare_metrics_json.py" in cmd and (
        "<scenario>" in cmd or "baselines/<scenario>" in cmd
    ):
        return ""
    if "run_holdout.py" in cmd and "--train-start" not in cmd:
        return default_holdout_command(run_id)
    return cmd


def coerce_eval_command(eval_command) -> list[str]:
    if not eval_command:
        return []
    if isinstance(eval_command, str):
        return [eval_command]
    if not isinstance(eval_command, list):
        return [str(eval_command)]
    tokens = [str(item) for item in eval_command if str(item).strip()]
    if not tokens:
        return []
    token_like = any(
        token.startswith("-") or "<RUN_DIR>" in token or "<CONFIG_PATH>" in token
        for token in tokens
    )
    if token_like:
        return [" ".join(tokens)]
    return tokens


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


def _read_json_or_none(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _derive_decision(comparison: dict | None) -> tuple[str, str, dict]:
    """
    Map compare_metrics_json decision to our publish/triage decision vocabulary.

    Returns: (decision, status_for_template, details)
      decision: accept | reject | iterate | needs-human
      status_for_template: pass | fail | inconclusive
    """

    if not comparison:
        return (
            "iterate",
            "inconclusive",
            {"source": "missing_comparison_json", "comparison_decision": None},
        )

    raw = str(comparison.get("decision") or "").strip().lower()
    details: dict = {"source": "comparison_json", "comparison_decision": raw or None}
    if isinstance(comparison.get("incomparable_reasons"), list):
        details["incomparable_reasons"] = comparison.get("incomparable_reasons")
    if isinstance(comparison.get("gates"), dict):
        details["gates"] = comparison.get("gates")

    if raw == "pass":
        return ("accept", "pass", details)
    if raw == "fail":
        return ("reject", "fail", details)
    if raw == "incomparable":
        # Treat as needs-human because baseline/candidate definitions differ.
        return ("needs-human", "inconclusive", details)
    return ("iterate", "inconclusive", details)


def _bundle_experiment_artifacts(
    *,
    root: Path,
    run_id: str,
    plan_path: Path,
    result_path: Path,
    log_path: Path,
    codex_log: Path,
    metrics_path: Path,
    comparison_path: Path | None,
    report_path: Path | None,
) -> None:
    out_dir = root / "artifacts" / "experiments" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Always include the minimal reproducibility bundle even if we don't open a PR.
    shutil.copy2(plan_path, out_dir / "plan.json")
    shutil.copy2(result_path, out_dir / "experiment_result.json")
    shutil.copy2(log_path, out_dir / "experiment.md")
    if codex_log.exists():
        shutil.copy2(codex_log, out_dir / "codex_implement.log")

    if metrics_path.exists():
        shutil.copy2(metrics_path, out_dir / "metrics.json")

    agents_path = root / "AGENTS.md"
    if agents_path.exists():
        shutil.copy2(agents_path, out_dir / "AGENTS.md")

    config_used_path = metrics_path.parent / "config_used.yaml"
    if not config_used_path.exists():
        metrics_json = (
            _read_json_or_none(metrics_path) if metrics_path.exists() else None
        )
        config_used_rel = (
            str(metrics_json.get("config_used_path") or "") if metrics_json else ""
        ).strip()
        if config_used_rel:
            config_used_path = root / config_used_rel
    if config_used_path.exists():
        shutil.copy2(config_used_path, out_dir / "config_used.yaml")

    if comparison_path is not None and comparison_path.exists():
        shutil.copy2(comparison_path, out_dir / "comparison.json")

    if report_path is not None and report_path.exists():
        shutil.copy2(report_path, out_dir / "backtest.md")


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


def ensure_codex_writable(log_path: Path) -> None:
    if not log_path.exists():
        return
    text = log_path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"^sandbox:\\s*(.+)$", text, flags=re.MULTILINE)
    if not match:
        return
    sandbox = match.group(1).strip().lower()
    if sandbox == "read-only":
        raise RuntimeError(
            "codex exec ran in read-only sandbox; cannot implement changes. "
            "Check codex config/profile and ensure agent_loop uses workspace-write "
            "or danger-full-access."
        )


def ensure_impl_changes(root: Path) -> None:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    if not status:
        raise RuntimeError(
            "No working tree changes after codex implementation. "
            "Ensure the prompt triggers real code/config edits and codex "
            "can write to the repo."
        )


def list_changed_paths(root: Path) -> list[str]:
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    if not status:
        return []
    paths = []
    for line in status.splitlines():
        path = line[3:].strip()
        if "->" in path:
            path = path.split("->")[-1].strip()
        paths.append(path.replace("\\", "/"))
    return paths


def ensure_no_disallowed_changes(root: Path) -> None:
    disallowed = [
        path
        for path in list_changed_paths(root)
        if path.startswith(("docs/experiments/", "tasks/"))
    ]
    if disallowed:
        raise RuntimeError(
            "Disallowed changes detected from codex implementation: "
            + ", ".join(disallowed)
        )


def ensure_substantive_changes(root: Path) -> None:
    substantive_prefixes = (
        "config/",
        "py64_analysis/",
        "py32_fetcher/",
        "scripts/",
        "tools/",
    )
    substantive = [
        path
        for path in list_changed_paths(root)
        if path.startswith(substantive_prefixes)
    ]
    if not substantive:
        raise RuntimeError(
            "No substantive code/config changes detected. "
            "Ensure the implementation updates config or source files, "
            "not only docs or task artifacts."
        )


def detect_experiment_config(root: Path) -> str | None:
    configs = [
        path
        for path in list_changed_paths(root)
        if path.startswith("config/experiments/") and path.endswith(".yaml")
    ]
    if not configs:
        return None
    if len(configs) > 1:
        raise RuntimeError(
            "Multiple experiment configs modified: " + ", ".join(configs)
        )
    return configs[0]


def render_experiment_log(
    template: Path, out_path: Path, plan: dict, result: dict
) -> None:
    text = template.read_text(encoding="utf-8")
    status = result.get("status", "inconclusive")
    next_action = "iterate"
    if status == "pass":
        next_action = "merge"
    elif status == "fail":
        next_action = "abandon"
    text = re.sub(
        r"(Experiment type:\s*)experiment\|infra",
        r"\1experiment",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(status:\s*)pass\|fail\|inconclusive",
        rf"\1{status}",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(next_action:\s*)merge\|iterate\|abandon",
        rf"\1{next_action}",
        text,
        flags=re.IGNORECASE,
    )
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
    prompt_text = prompt_path.read_text(encoding="utf-8-sig")
    payload = json.dumps(plan, ensure_ascii=True)
    cmd = [
        codex_bin,
        "exec",
        "--profile",
        profile,
        "--dangerously-bypass-approvals-and-sandbox",
        "-",
    ]
    prompt_payload = prompt_text + "\n\nPLAN_JSON:\n" + payload
    with open(log_path, "w", encoding="utf-8") as log:
        result = subprocess.run(
            cmd,
            cwd=root,
            stdout=log,
            stderr=log,
            text=True,
            encoding="utf-8",
            input=prompt_payload,
        )
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
        reason = plan.get("reason", "").strip()
        if reason:
            reason = f"{reason} (overridden to do)"
        else:
            reason = "overridden to do"
        plan["decision"] = "do"
        plan["reason"] = reason
    run_id = plan.get("run_id", "")
    if not run_id:
        raise RuntimeError("plan.run_id is required.")

    branch = f"agent/{run_id}-{slugify(plan['title'])}"
    run(["git", "checkout", "-b", branch], cwd=root)

    log_dir = root / "artifacts" / "agent"
    log_dir.mkdir(parents=True, exist_ok=True)
    codex_log = log_dir / f"{plan['run_id']}_implement.log"
    run_codex(root, root / args.prompt, plan, args.profile, codex_log)
    ensure_codex_writable(codex_log)
    ensure_impl_changes(root)
    ensure_no_disallowed_changes(root)
    ensure_substantive_changes(root)

    eval_cmd = coerce_eval_command(plan.get("eval_command"))
    if not eval_cmd:
        raise RuntimeError("eval_command is empty.")
    normalized_cmds = [normalize_eval_command(cmd, run_id) for cmd in eval_cmd]
    normalized_cmds = [cmd for cmd in normalized_cmds if cmd]
    if not normalized_cmds:
        normalized_cmds = [default_holdout_command(run_id)]
    config_path = detect_experiment_config(root)
    if config_path:
        normalized_cmds = [f"$env:KEIBA_CONFIG_PATH='{config_path}'"] + normalized_cmds
    run_shell_commands(normalized_cmds, cwd=root)

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

    decision, status, decision_details = _derive_decision(
        _read_json_or_none(comparison_candidate)
    )

    result = {
        "run_id": run_id,
        "seed_id": plan["seed_id"],
        "title": plan["title"],
        "status": status,
        "decision": {
            "decision": decision,
            "status": status,
            "details": decision_details,
        },
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

    _bundle_experiment_artifacts(
        root=root,
        run_id=run_id,
        plan_path=plan_path,
        result_path=result_path,
        log_path=log_path,
        codex_log=codex_log,
        metrics_path=metrics_path,
        comparison_path=comparison_candidate if comparison_candidate.exists() else None,
        report_path=report_candidate if report_candidate.exists() else None,
    )

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
