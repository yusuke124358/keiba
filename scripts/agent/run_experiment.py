#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


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


def select_holdout_tokens(eval_cmds: list[str], run_id: str) -> list[str]:
    raw = ""
    for cmd in eval_cmds:
        cmd2 = normalize_eval_command(cmd, run_id)
        if cmd2 and "run_holdout.py" in cmd2:
            raw = cmd2
            break
    if not raw:
        raw = default_holdout_command(run_id)

    tokens = raw.strip().split()
    if not tokens:
        raise RuntimeError("eval_command is empty after normalization.")

    first = Path(tokens[0]).name.lower()
    if first in {"python", "python3", "py"} or first.endswith("python.exe"):
        tokens = tokens[1:]

    if not tokens or not tokens[0].endswith("run_holdout.py"):
        raise RuntimeError(
            "Only run_holdout.py eval_command is supported for statistical outputs. "
            f"Got: {raw}"
        )
    return tokens


def compute_baseline_key(
    base_commit: str, test_start: str, test_end: str, config_hash: str | None
) -> str:
    payload = {
        "base_commit": base_commit,
        "test_start": test_start,
        "test_end": test_end,
        "config_hash": config_hash or "unknown",
    }
    blob = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    return _sha256_bytes(blob)


def run_holdout(
    root: Path,
    base_tokens: list[str],
    *,
    name: str,
    out_dir: Path,
    config_path: str | None,
) -> None:
    tokens = list(base_tokens)
    _set_arg(tokens, "--name", name)
    _set_arg(tokens, "--out-dir", str(out_dir).replace("\\", "/"))

    env = os.environ.copy()
    # Prevent runner-wide env leaking into baseline/variant runs.
    env["KEIBA_CONFIG_PATH"] = config_path or ""

    result = subprocess.run([sys.executable, *tokens], cwd=root, env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join([sys.executable, *tokens])}"
        )


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


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _rel(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def _parse_arg(tokens: list[str], flag: str) -> str | None:
    if flag in tokens:
        i = tokens.index(flag)
        return tokens[i + 1] if i + 1 < len(tokens) else None
    prefix = flag + "="
    for t in tokens:
        if t.startswith(prefix):
            return t[len(prefix) :]
    return None


def _set_arg(tokens: list[str], flag: str, value: str) -> None:
    if flag in tokens:
        i = tokens.index(flag)
        if i + 1 < len(tokens):
            tokens[i + 1] = value
            return
        tokens.append(value)
        return
    prefix = flag + "="
    for i, t in enumerate(tokens):
        if t.startswith(prefix):
            tokens[i] = f"{flag}={value}"
            return
    tokens.extend([flag, value])


def resolve_holdout_config_path(
    root: Path, holdout_tokens: list[str], run_id: str
) -> Path:
    """
    Resolve the effective config file for `run_holdout.py` given our execution model.

    `run_holdout.py` resolves config in this order:
    - explicit `--config` (or `$env:KEIBA_CONFIG_PATH`)
    - else auto: `config/experiments/{--name}.yaml|yml` when KEIBA_CONFIG_PATH is unset/empty
    - else default `config/config.yaml`

    In this agent runner we intentionally override `KEIBA_CONFIG_PATH` to avoid
    leaking runner-wide env into evaluations. So config resolution is
    deterministic from repo files + CLI args.
    """

    cli_cfg = _parse_arg(holdout_tokens, "--config")
    if cli_cfg:
        p = Path(cli_cfg)
        return p if p.is_absolute() else (root / p)

    cand = root / "config" / "experiments" / f"{run_id}.yaml"
    if cand.exists():
        return cand
    cand = root / "config" / "experiments" / f"{run_id}.yml"
    if cand.exists():
        return cand

    return root / "config" / "config.yaml"


def _format_float(x: Any) -> str:
    if x is None:
        return "(missing)"
    try:
        return f"{float(x):.6f}"
    except Exception:
        s = str(x)
        return "(missing)" if s == "None" else s


def _format_int(x: Any) -> str:
    if x is None:
        return "(missing)"
    try:
        return str(int(x))
    except Exception:
        s = str(x)
        return "(missing)" if s == "None" else s


def _format_ci95(x: Any) -> str:
    if not isinstance(x, (list, tuple)) or len(x) != 2:
        return "(missing)"
    return f"[{_format_float(x[0])}, {_format_float(x[1])}]"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_metrics_for_log(metrics_json: dict[str, Any]) -> dict[str, Any]:
    backtest = metrics_json.get("backtest")
    if not isinstance(backtest, dict):
        raise RuntimeError("metrics.json missing required 'backtest' section.")

    split = (
        metrics_json.get("split", {})
        if isinstance(metrics_json.get("split"), dict)
        else {}
    )
    test_split = split.get("test", {}) if isinstance(split.get("test"), dict) else {}
    test_start = str(test_split.get("start") or "N/A")
    test_end = str(test_split.get("end") or "N/A")
    test_period = f"{test_start} to {test_end}"

    roi = backtest.get("roi")
    total_stake = backtest.get("total_stake")
    total_profit = backtest.get("total_profit")
    total_return = None
    try:
        if total_stake is not None and total_profit is not None:
            total_return = float(total_stake) + float(total_profit)
    except Exception:
        total_return = None

    run_kind = str(metrics_json.get("run_kind") or "holdout")
    rolling = "yes" if run_kind == "rolling_holdout" else "no"

    step14 = metrics_json.get("step14") or {}
    step14_roi = step14.get("roi") if isinstance(step14, dict) else None
    pooled_vs_step14_mismatch = "no"
    preferred_roi = "pooled"
    if rolling == "yes" and step14_roi is not None and roi is not None:
        try:
            if (float(roi) >= 0) != (float(step14_roi) >= 0):
                pooled_vs_step14_mismatch = "yes"
            preferred_roi = "step14"
        except Exception:
            preferred_roi = "step14"

    return {
        "roi": float(roi) if roi is not None else None,
        "total_stake": float(total_stake) if total_stake is not None else None,
        "total_return": float(total_return) if total_return is not None else None,
        "n_bets": int(backtest.get("n_bets"))
        if backtest.get("n_bets") is not None
        else None,
        "max_drawdown": float(backtest.get("max_drawdown"))
        if backtest.get("max_drawdown") is not None
        else None,
        "test_period": test_period,
        "test_start": test_start,
        "test_end": test_end,
        "rolling": rolling,
        "design_window": "N/A",
        "eval_window": "N/A",
        "paired_delta": "N/A",
        "pooled_vs_step14_mismatch": pooled_vs_step14_mismatch,
        "preferred_roi": preferred_roi,
        "config_hash_sha256": metrics_json.get("config_hash_sha256"),
        "config_used_path": metrics_json.get("config_used_path"),
        "data_cutoff": metrics_json.get("data_cutoff") or {},
        "run_kind": run_kind,
    }


def compute_candidate_scarcity(
    run_dir: Path,
    *,
    candidate_threshold: int = 1,
) -> tuple[float | None, float | None]:
    """
    Compute simple candidate scarcity indicators from run artifacts.

    We define "day" as a race day present in `race_ids_test.txt` (not calendar days).
    Since we do not persist full candidate lists, we treat "candidates" as
    executed bets (stake > 0) for the purpose of `frac_days_candidates_ge_n`.
    """

    def _race_id_to_day(s: str) -> str | None:
        s = (s or "").strip()
        if len(s) >= 8 and s[:8].isdigit():
            return s[:8]
        return None

    total_days: int | None = None
    race_ids_path = run_dir / "race_ids_test.txt"
    if race_ids_path.exists():
        days: set[str] = set()
        for line in race_ids_path.read_text(encoding="utf-8").splitlines():
            day = _race_id_to_day(line)
            if day:
                days.add(day)
        total_days = len(days)

    bets_csv = run_dir / "bets.csv"
    if not bets_csv.exists():
        return None, None

    try:
        import pandas as pd

        df = pd.read_csv(bets_csv)
        if df.empty or "race_id" not in df.columns:
            if total_days is None or total_days <= 0:
                return None, None
            return 0.0, 0.0

        df["_day"] = df["race_id"].astype(str).str.slice(0, 8)
        stake = pd.to_numeric(df.get("stake"), errors="coerce").fillna(0.0)
        df["_has_bet"] = stake > 0

        bet_counts = df[df["_has_bet"]].groupby("_day").size()
        n_days_any_bet = int(len(bet_counts))

        threshold_n = int(candidate_threshold)
        if "daily_top_n_n" in df.columns:
            vals = pd.to_numeric(df["daily_top_n_n"], errors="coerce").dropna().unique()
            if len(vals) == 1:
                try:
                    n_val = int(vals[0])
                    if n_val > 0:
                        threshold_n = n_val
                except Exception:
                    pass

        n_days_ge_n = int((bet_counts >= threshold_n).sum()) if threshold_n > 0 else 0

        denom = total_days if total_days is not None and total_days > 0 else None
        if denom is None:
            denom = int(df["_day"].nunique())
        if denom <= 0:
            return None, None

        frac_days_any_bet = float(n_days_any_bet / denom)
        frac_days_candidates_ge_n = float(n_days_ge_n / denom)
        return frac_days_candidates_ge_n, frac_days_any_bet
    except Exception:
        return None, None


def propose_decision(
    *,
    delta_roi_ci95: list[float],
    delta_max_drawdown: float | None,
    robustness: list[dict[str, Any]],
    leakage_pre_race_only: str,
) -> tuple[str, str, list[str]]:
    flags: list[str] = []
    lo, hi = float(delta_roi_ci95[0]), float(delta_roi_ci95[1])
    if lo <= 0 <= hi:
        flags.append("CI_crosses_0")
    if delta_max_drawdown is not None and delta_max_drawdown > 0:
        flags.append("drawdown_worse")

    neg = 0
    tot = 0
    for row in robustness:
        v = row.get("roi")
        b = row.get("baseline_roi")
        vs = row.get("stake", 0.0)
        bs = row.get("baseline_stake", 0.0)
        if v is None or b is None:
            continue
        try:
            if float(vs) <= 0 or float(bs) <= 0:
                continue
            tot += 1
            if float(v) - float(b) < 0:
                neg += 1
        except Exception:
            continue
    if tot > 0 and neg / tot > 0.5:
        flags.append("robustness_worse_majority")

    if leakage_pre_race_only == "no":
        flags.append("leakage_check_failed_or_unknown")
        return "needs-human", "Leakage safety check did not pass (heuristic).", flags

    if hi < 0:
        return "reject", "delta_ROI_CI95.upper < 0", flags

    accept_drawdown_ok = delta_max_drawdown is None or delta_max_drawdown <= 0
    accept_robust_ok = tot == 0 or neg / tot <= 0.5
    if lo > 0 and accept_drawdown_ok and accept_robust_ok:
        return (
            "accept",
            "delta_ROI_CI95.lower > 0 and no robustness/drawdown regression.",
            flags,
        )

    return (
        "iterate",
        "Uncertain improvement (CI crosses 0 and/or robustness/drawdown unclear).",
        flags,
    )


PLACEHOLDER_RE = re.compile(r"<[^>]+>")


def ensure_no_placeholders(text: str) -> None:
    bad: list[str] = []
    for m in PLACEHOLDER_RE.finditer(text):
        bad.append(m.group(0))
        if len(bad) >= 5:
            break
    if bad:
        raise RuntimeError(
            "Experiment log contains template placeholders: " + ", ".join(bad)
        )


def validate_schema(result: dict[str, Any], schema_path: Path) -> None:
    try:
        import jsonschema  # type: ignore[import-untyped]
    except Exception as exc:
        raise RuntimeError(
            "jsonschema is required for experiment_result validation. "
            "Install with: pip install jsonschema"
        ) from exc

    schema = json.loads(schema_path.read_text(encoding="utf-8-sig"))
    jsonschema.validate(instance=result, schema=schema)


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


def render_experiment_md(
    *,
    run_id: str,
    plan: dict[str, Any],
    base_commit: str,
    head_commit: str,
    variant_metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    stats: dict[str, Any],
    breakdowns: dict[str, Any],
    decision: dict[str, Any],
    artifacts: dict[str, Any],
    holdout_tokens: list[str],
    bootstrap_seed: int,
) -> str:
    title = str(plan.get("title") or "").strip()
    hypothesis = str(plan.get("hypothesis") or "").strip()
    risk_level = str(plan.get("risk_level") or "medium").strip()
    max_diff_size = int(plan.get("max_diff_size") or 0)

    test_period = str(variant_metrics.get("test_period") or "N/A")
    eval_cmd = " ".join([sys.executable, *holdout_tokens]).replace("\\", "/")

    changed_files = artifacts.get("changed_files") or []
    change_summary = "(unavailable)"
    if changed_files:
        items = ", ".join(changed_files[:6])
        suffix = "" if len(changed_files) <= 6 else f" (+{len(changed_files) - 6} more)"
        change_summary = f"{items}{suffix}"

    # Robustness tables.
    odds_rows = breakdowns.get("odds_bucket") or []
    odds_table = [
        "| odds_bucket | ROI | stake | bets | baseline_ROI | baseline_stake | baseline_bets |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in odds_rows:
        odds_table.append(
            "| {bucket} | {roi} | {stake} | {bets} | {b_roi} | {b_stake} | {b_bets} |".format(
                bucket=r.get("odds_bucket"),
                roi="N/A" if r.get("roi") is None else _format_float(r.get("roi")),
                stake=_format_float(r.get("stake")),
                bets=int(r.get("bets") or 0),
                b_roi="N/A"
                if r.get("baseline_roi") is None
                else _format_float(r.get("baseline_roi")),
                b_stake=_format_float(r.get("baseline_stake")),
                b_bets=int(r.get("baseline_bets") or 0),
            )
        )

    month_rows = breakdowns.get("month") or []
    month_table = [
        "| month | ROI | stake | bets | baseline_ROI | baseline_stake | baseline_bets |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in month_rows:
        month_table.append(
            "| {month} | {roi} | {stake} | {bets} | {b_roi} | {b_stake} | {b_bets} |".format(
                month=r.get("month"),
                roi="N/A" if r.get("roi") is None else _format_float(r.get("roi")),
                stake=_format_float(r.get("stake")),
                bets=int(r.get("bets") or 0),
                b_roi="N/A"
                if r.get("baseline_roi") is None
                else _format_float(r.get("baseline_roi")),
                b_stake=_format_float(r.get("baseline_stake")),
                b_bets=int(r.get("baseline_bets") or 0),
            )
        )

    lines: list[str] = []
    lines.append(f"# Experiment {run_id} - {title}")
    lines.append("")
    lines.append("## Metadata")
    lines.append(f"- run_id: {run_id}")
    lines.append(f"- seed_id: {plan.get('seed_id')}")
    lines.append(f"- base_commit: {base_commit}")
    lines.append(f"- head_commit: {head_commit}")
    lines.append(f"- python_executable: {sys.executable}")
    lines.append(f"- bootstrap_seed: {bootstrap_seed}")
    lines.append("- unit_of_resampling: day-block (YYYY-MM-DD)")
    lines.append("")
    lines.append("## Hypothesis")
    lines.append(hypothesis or "(missing)")
    lines.append("")
    lines.append("## Experimental Design")
    lines.append(f"- test_period: {test_period}")
    lines.append(f"- eval_command: `{eval_cmd}`")
    lines.append(
        f"- config_used (variant): {artifacts.get('config_used_variant', 'N/A')}"
    )
    lines.append(
        f"- config_used (baseline): {artifacts.get('config_used_baseline', 'N/A')}"
    )
    lines.append("")
    lines.append("## Change Summary")
    lines.append(f"Changed files: {change_summary}.")
    lines.append("")
    lines.append("## Risk")
    lines.append("- Experiment type: experiment")
    lines.append(f"- risk_level: {risk_level}")
    lines.append(f"- max_diff_size: {max_diff_size}")
    lines.append("")
    lines.append("## Metrics (required)")
    lines.append(f"- ROI: {_format_float(variant_metrics.get('roi'))}")
    lines.append(f"- Total stake: {_format_float(variant_metrics.get('total_stake'))}")
    lines.append(f"- n_bets: {_format_int(variant_metrics.get('n_bets'))}")
    lines.append(f"- Test period: {test_period}")
    lines.append(
        f"- Max drawdown: {_format_float(variant_metrics.get('max_drawdown'))}"
    )
    lines.append(
        f"- frac_days_any_bet: {_format_float(variant_metrics.get('frac_days_any_bet'))}"
    )
    lines.append(
        f"- frac_days_candidates_ge_n: {_format_float(variant_metrics.get('frac_days_candidates_ge_n'))}"
    )
    lines.append("- ROI definition: ROI = profit / stake, profit = return - stake.")
    lines.append(f"- Rolling: {variant_metrics.get('rolling') or 'no'}")
    lines.append(f"- Design window: {variant_metrics.get('design_window') or 'N/A'}")
    lines.append(f"- Eval window: {variant_metrics.get('eval_window') or 'N/A'}")
    lines.append(
        f"- Paired delta vs baseline: {variant_metrics.get('paired_delta') or 'N/A'}"
    )
    lines.append(
        f"- Pooled vs step14 sign mismatch: {variant_metrics.get('pooled_vs_step14_mismatch') or 'no'}"
    )
    lines.append(
        f"- Preferred ROI for decisions: {variant_metrics.get('preferred_roi') or 'pooled'}"
    )
    lines.append("")
    lines.append("## Baseline (point estimates)")
    lines.append(f"- baseline_ROI: {_format_float(baseline_metrics.get('roi'))}")
    lines.append(
        f"- baseline_total_stake: {_format_float(baseline_metrics.get('total_stake'))}"
    )
    lines.append(f"- baseline_n_bets: {_format_int(baseline_metrics.get('n_bets'))}")
    lines.append(
        f"- baseline_max_drawdown: {_format_float(baseline_metrics.get('max_drawdown'))}"
    )
    lines.append(
        f"- baseline_frac_days_any_bet: {_format_float(baseline_metrics.get('frac_days_any_bet'))}"
    )
    lines.append(
        f"- baseline_frac_days_candidates_ge_n: {_format_float(baseline_metrics.get('frac_days_candidates_ge_n'))}"
    )
    lines.append("")
    lines.append("## Paired Delta vs Baseline")
    lines.append(f"- delta_ROI: {_format_float(decision.get('delta_roi'))}")
    lines.append(
        f"- delta_max_drawdown: {_format_float(decision.get('delta_max_drawdown'))}"
    )
    lines.append("")
    lines.append("## Uncertainty (day-block bootstrap)")
    lines.append(f"- ROI_variant_CI95: {_format_ci95(stats.get('roi_ci95'))}")
    lines.append(f"- ROI_baseline_CI95: {_format_ci95(stats.get('baseline_roi_ci95'))}")
    lines.append(f"- delta_ROI_CI95: {_format_ci95(stats.get('delta_roi_ci95'))}")
    lines.append(
        f"- p_one_sided (P(delta<=0)): {_format_float(stats.get('p_one_sided_delta_le_0'))}"
    )
    lines.append(f"- bootstrap: {stats.get('bootstrap') or {}}")
    lines.append("")
    lines.append("## Robustness")
    lines.append("### ROI by odds bucket")
    lines.extend(odds_table)
    lines.append("")
    lines.append("### ROI by month")
    lines.extend(month_table)
    lines.append("")
    lines.append("## Artifacts")
    for k, v in artifacts.items():
        if k == "changed_files":
            continue
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Decision")
    lines.append(f"- decision: {decision.get('decision') or '(missing)'}")
    lines.append(f"- rationale: {decision.get('rationale') or '(missing)'}")
    lines.append(f"- flags: {', '.join(decision.get('flags') or []) or '(none)'}")
    lines.append("")
    lines.append("## Notes")
    lines.append(
        f"- leakage_check_pre_race_only: {decision.get('leakage_pre_race_only') or 'unknown'}"
    )

    return "\n".join(lines).rstrip() + "\n"


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
    p.add_argument("--bootstrap-b", type=int, default=2000)
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

    base_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()

    # Baseline (cacheable): run holdout on base_commit before codex changes.
    eval_cmds = coerce_eval_command(plan.get("eval_command"))
    holdout_tokens = select_holdout_tokens(eval_cmds, run_id)
    test_start = _parse_arg(holdout_tokens, "--test-start")
    test_end = _parse_arg(holdout_tokens, "--test-end")
    if not test_start or not test_end:
        raise RuntimeError("eval_command must include --test-start and --test-end.")

    baseline_cfg_path = resolve_holdout_config_path(root, holdout_tokens, run_id)
    baseline_cfg_hash = _sha256_file(baseline_cfg_path)
    baseline_key = compute_baseline_key(
        base_commit, test_start, test_end, baseline_cfg_hash
    )
    baseline_dir = root / "artifacts" / "baselines" / baseline_key
    baseline_metrics_path = baseline_dir / "metrics.json"
    baseline_pnl_path = baseline_dir / "per_bet_pnl.csv"

    if not (baseline_metrics_path.exists() and baseline_pnl_path.exists()):
        print(f"[baseline] cache miss -> running baseline in {baseline_dir}")
        run_holdout(
            root,
            holdout_tokens,
            # Keep `--name` aligned with variant so run_holdout.py auto-config stays consistent.
            name=run_id,
            out_dir=baseline_dir,
            config_path=None,
        )
    else:
        print(f"[baseline] cache hit: {baseline_dir}")

    baseline_extracted = extract_metrics_for_log(load_json(baseline_metrics_path))

    log_dir = root / "artifacts" / "agent"
    log_dir.mkdir(parents=True, exist_ok=True)
    codex_log = log_dir / f"{plan['run_id']}_implement.log"
    run_codex(root, root / args.prompt, plan, args.profile, codex_log)
    ensure_codex_writable(codex_log)
    ensure_impl_changes(root)
    ensure_no_disallowed_changes(root)
    ensure_substantive_changes(root)
    exp_config_path = detect_experiment_config(root)

    # Commit code changes before evaluation so head_commit is stable/reproducible.
    run(["git", "add", "-A"], cwd=root)
    run(["git", "commit", "-m", f"agent: {run_id} {slugify(plan['title'])}"], cwd=root)
    head_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    changed_files = subprocess.run(
        ["git", "diff", "--name-only", f"{base_commit}..{head_commit}"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.splitlines()
    changed_files = [p.strip() for p in changed_files if p.strip()]

    # Variant evaluation (uses experiment config if one was modified).
    run_dir = root / "data" / "holdout_runs" / run_id
    print(f"[variant] running eval into {run_dir}")
    run_holdout(
        root,
        holdout_tokens,
        name=run_id,
        out_dir=run_dir,
        config_path=exp_config_path,
    )

    variant_metrics_path = run_dir / "metrics.json"
    variant_pnl_path = run_dir / "per_bet_pnl.csv"
    if not variant_metrics_path.exists():
        raise RuntimeError(f"metrics.json not found: {variant_metrics_path}")
    if not variant_pnl_path.exists():
        raise RuntimeError(f"per_bet_pnl.csv not found: {variant_pnl_path}")

    variant_extracted = extract_metrics_for_log(load_json(variant_metrics_path))

    v_frac_ge_n, v_frac_any_bet = compute_candidate_scarcity(run_dir)
    b_frac_ge_n, b_frac_any_bet = compute_candidate_scarcity(baseline_dir)
    variant_extracted["frac_days_candidates_ge_n"] = v_frac_ge_n
    variant_extracted["frac_days_any_bet"] = v_frac_any_bet
    baseline_extracted["frac_days_candidates_ge_n"] = b_frac_ge_n
    baseline_extracted["frac_days_any_bet"] = b_frac_any_bet

    exp_art_dir = root / "artifacts" / "experiments" / run_id
    exp_art_dir.mkdir(parents=True, exist_ok=True)
    variant_pnl_art = exp_art_dir / "per_bet_pnl.csv"
    variant_pnl_art.write_bytes(variant_pnl_path.read_bytes())

    # Ensure the experiment artifact bundle is self-contained for audits/repro.
    agents_path = root / "AGENTS.md"
    if agents_path.exists():
        shutil.copy2(agents_path, exp_art_dir / "AGENTS.md")
    variant_config_used = run_dir / "config_used.yaml"
    if variant_config_used.exists():
        shutil.copy2(variant_config_used, exp_art_dir / "config_used_variant.yaml")
    baseline_config_used = baseline_dir / "config_used.yaml"
    if baseline_config_used.exists():
        shutil.copy2(baseline_config_used, exp_art_dir / "config_used_baseline.yaml")

    bootstrap_seed = int(_sha256_bytes(run_id.encode("utf-8"))[:8], 16)
    summary_stats_path = exp_art_dir / "summary_stats.json"
    run(
        [
            sys.executable,
            "scripts/stats/block_bootstrap.py",
            "--variant",
            str(variant_pnl_art),
            "--baseline",
            str(baseline_pnl_path),
            "--out",
            str(summary_stats_path),
            "--B",
            str(int(args.bootstrap_b)),
            "--seed",
            str(bootstrap_seed),
        ],
        cwd=root,
    )
    summary_stats = load_json(summary_stats_path)
    stats = summary_stats["stats"]
    breakdowns = summary_stats["breakdowns"]

    delta_roi = None
    if variant_extracted["roi"] is not None and baseline_extracted["roi"] is not None:
        delta_roi = float(variant_extracted["roi"]) - float(baseline_extracted["roi"])
    delta_dd = None
    if (
        variant_extracted["max_drawdown"] is not None
        and baseline_extracted["max_drawdown"] is not None
    ):
        delta_dd = float(variant_extracted["max_drawdown"]) - float(
            baseline_extracted["max_drawdown"]
        )

    if delta_roi is not None:
        variant_extracted["paired_delta"] = _format_float(delta_roi)

    leakage_pre_race_only = "yes"
    if any(p.startswith("py64_analysis/src/keiba/features/") for p in changed_files):
        leakage_pre_race_only = "no"

    decision_value, rationale, flags = propose_decision(
        delta_roi_ci95=stats["delta_roi_ci95"],
        delta_max_drawdown=delta_dd,
        robustness=breakdowns.get("odds_bucket") or [],
        leakage_pre_race_only=leakage_pre_race_only,
    )
    status = "inconclusive"
    if decision_value == "accept":
        status = "pass"
    elif decision_value == "reject":
        status = "fail"

    artifacts = {
        "metrics_json": _rel(root, variant_metrics_path),
        "baseline_metrics_json": _rel(root, baseline_metrics_path),
        "per_bet_pnl": _rel(root, variant_pnl_art),
        "baseline_per_bet_pnl": _rel(root, baseline_pnl_path),
        "summary_stats_json": _rel(root, summary_stats_path),
        "report": _rel(root, run_dir / "report" / "backtest.md")
        if (run_dir / "report" / "backtest.md").exists()
        else "N/A",
        "config_used_variant": _rel(root, run_dir / "config_used.yaml")
        if (run_dir / "config_used.yaml").exists()
        else "N/A",
        "config_used_baseline": _rel(root, baseline_dir / "config_used.yaml")
        if (baseline_dir / "config_used.yaml").exists()
        else "N/A",
        "changed_files": changed_files,
    }

    result = {
        "run_id": run_id,
        "seed_id": plan["seed_id"],
        "title": plan["title"],
        "status": status,
        "metrics": {
            "roi": variant_extracted["roi"],
            "total_stake": variant_extracted["total_stake"],
            "total_return": variant_extracted["total_return"],
            "n_bets": variant_extracted["n_bets"],
            "n_races": summary_stats["variant"]["n_races"],
            "n_days": summary_stats["variant"]["n_days"],
            "max_drawdown": variant_extracted["max_drawdown"],
            "test_period": variant_extracted["test_period"],
            "frac_days_candidates_ge_n": variant_extracted.get(
                "frac_days_candidates_ge_n"
            ),
            "frac_days_any_bet": variant_extracted.get("frac_days_any_bet"),
            "rolling": variant_extracted["rolling"],
            "design_window": variant_extracted["design_window"],
            "eval_window": variant_extracted["eval_window"],
            "paired_delta": variant_extracted["paired_delta"],
            "pooled_vs_step14_mismatch": variant_extracted["pooled_vs_step14_mismatch"],
            "preferred_roi": variant_extracted["preferred_roi"],
        },
        "baseline_metrics": {
            "roi": baseline_extracted["roi"],
            "total_stake": baseline_extracted["total_stake"],
            "total_return": baseline_extracted["total_return"],
            "n_bets": baseline_extracted["n_bets"],
            "n_races": summary_stats["baseline"]["n_races"],
            "n_days": summary_stats["baseline"]["n_days"],
            "max_drawdown": baseline_extracted["max_drawdown"],
            "test_period": baseline_extracted["test_period"],
            "frac_days_candidates_ge_n": baseline_extracted.get(
                "frac_days_candidates_ge_n"
            ),
            "frac_days_any_bet": baseline_extracted.get("frac_days_any_bet"),
        },
        "deltas": {"delta_roi": delta_roi, "delta_max_drawdown": delta_dd},
        "stats": {
            "roi_ci95": stats["roi_ci95"],
            "baseline_roi_ci95": stats["baseline_roi_ci95"],
            "delta_roi_ci95": stats["delta_roi_ci95"],
            "p_one_sided_delta_le_0": stats["p_one_sided_delta_le_0"],
            "bootstrap": stats["bootstrap"],
            "resampling_unit": stats.get("resampling_unit"),
        },
        "breakdowns": breakdowns,
        "reproducibility": {
            "base_commit": base_commit,
            "head_commit": head_commit,
            "config_hash": variant_extracted.get("config_hash_sha256"),
            "eval_command": " ".join([sys.executable, *holdout_tokens]).replace(
                "\\", "/"
            ),
            "python_executable": sys.executable,
            "environment_summary": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
                "runner_os": os.environ.get("RUNNER_OS"),
            },
            "bootstrap_seed": bootstrap_seed,
        },
        "decision": {
            "decision": decision_value,
            "rationale": rationale,
            "flags": flags,
            "leakage_pre_race_only": leakage_pre_race_only,
        },
        "artifacts": artifacts,
    }

    result_path = root / "experiments" / "runs" / f"{plan['run_id']}.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=True, indent=2), encoding="utf-8"
    )

    md_text = render_experiment_md(
        run_id=run_id,
        plan=plan,
        base_commit=base_commit,
        head_commit=head_commit,
        variant_metrics=variant_extracted,
        baseline_metrics=baseline_extracted,
        stats=stats,
        breakdowns=breakdowns,
        decision={
            "decision": decision_value,
            "rationale": rationale,
            "flags": flags,
            "delta_roi": delta_roi,
            "delta_max_drawdown": delta_dd,
            "leakage_pre_race_only": leakage_pre_race_only,
        },
        artifacts=artifacts,
        holdout_tokens=holdout_tokens,
        bootstrap_seed=bootstrap_seed,
    )
    ensure_no_placeholders(md_text)

    log_path = root / "docs" / "experiments" / f"{plan['run_id']}.md"
    log_path.write_text(md_text, encoding="utf-8")

    validate_schema(result, root / args.metrics_schema)

    run(["git", "add", str(result_path), str(log_path)], cwd=root)
    run(["git", "commit", "-m", f"agent: {plan['run_id']} results"], cwd=root)
    print(f"Committed {plan['run_id']} on {branch}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
