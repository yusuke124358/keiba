"""Audit stake-odds-damp replay vs engine outputs on a single window."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import shutil
import hashlib
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml


STAKE_COLS = ["stake", "stake_yen", "stake_amount", "bet_amount", "stake_raw", "stake_after"]
RETURN_COLS = ["return", "return_yen", "payout", "payout_yen"]
PROFIT_COLS = ["profit", "net_profit", "pnl"]
ODDS_COLS = ["odds_at_buy", "odds", "buy_odds", "win_odds"]
KEY_COL_CANDIDATES = [
    ["race_id", "horse_no", "asof_time"],
    ["race_id", "horse_id", "asof_time"],
    ["race_id", "horse_no"],
    ["race_id", "horse_id"],
]


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _find_col(cols: list[str], candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in cols:
            return c
    return None


def _detect_key_cols(cols: list[str]) -> Optional[list[str]]:
    for cand in KEY_COL_CANDIDATES:
        if all(c in cols for c in cand):
            return cand
    return None


def _hash_rows(df: pd.DataFrame, key_cols: list[str], stake_col: str, odds_col: str, extra_cols: list[str]) -> str:
    use_cols = key_cols + [stake_col, odds_col] + extra_cols
    use_cols = [c for c in use_cols if c in df.columns]
    safe = df[use_cols].copy()
    for col in use_cols:
        if col in key_cols:
            safe[col] = safe[col].astype(str)
        else:
            safe[col] = pd.to_numeric(safe[col], errors="coerce").fillna(0.0).map(lambda v: f"{v:.6f}")
    joined = safe.apply(lambda r: "|".join(r.values.tolist()), axis=1).tolist()
    joined = "\n".join(sorted(joined))
    return str(hashlib.sha1(joined.encode("utf-8")).hexdigest())


def _apply_stake_damp(
    df: pd.DataFrame,
    *,
    stake_col: str,
    odds_col: str,
    ref_odds: float,
    power: float,
    min_mult: float,
    min_yen: int,
) -> pd.DataFrame:
    out = df.copy()
    stake = pd.to_numeric(out[stake_col], errors="coerce").fillna(0.0)
    odds = pd.to_numeric(out[odds_col], errors="coerce")
    try:
        power = float(power)
    except Exception:
        power = 1.0
    if not np.isfinite(power) or power <= 0:
        power = 1.0
    try:
        min_mult = float(min_mult)
    except Exception:
        min_mult = 0.0
    if not np.isfinite(min_mult):
        min_mult = 0.0
    min_mult = min(1.0, max(0.0, min_mult))
    unit = int(min_yen) if min_yen and min_yen > 0 else 1
    if unit <= 0:
        unit = 1

    ratio = pd.Series([np.nan] * len(out), index=out.index)
    mult_raw = pd.Series([1.0] * len(out), index=out.index)
    mult = mult_raw.copy()
    stake_after = stake.copy()
    if ref_odds > 0:
        valid = (odds > 0) & np.isfinite(odds)
        ratio = odds / float(ref_odds)
        ratio = ratio.where(valid, np.nan)
        low_mask = ratio < 1.0
        raw = pd.Series(np.power(ratio, power), index=out.index)
        raw = raw.where(np.isfinite(raw) & (raw > 0), 0.0)
        raw = raw.where(low_mask, 1.0)
        mult_raw = raw.where(valid, 1.0)
        mult = np.maximum(min_mult, mult_raw)
        mult = np.minimum(1.0, mult)
        mult = pd.Series(mult, index=out.index)
        fallback = max(float(min_mult), 1e-9)
        mult = mult.where(~(valid & low_mask & (mult <= 0)), fallback)
        mult = mult.where(valid, 1.0)
        stake_raw = stake * mult
        stake_floor = (stake_raw // unit) * unit if unit > 0 else stake_raw
        if unit > 0:
            stake_floor = stake_floor.where(stake <= 0, np.maximum(unit, stake_floor))
        stake_floor = stake_floor.where(stake_floor <= stake, stake)
        stake_after = stake_after.where(~low_mask, stake_floor)

    mult_used = stake_after / stake.replace({0.0: np.nan})
    mult_used = mult_used.fillna(0.0)

    out[stake_col] = stake_after
    ret_cols = [c for c in RETURN_COLS if c in out.columns]
    if ret_cols:
        for rc in ret_cols:
            out[rc] = pd.to_numeric(out[rc], errors="coerce").fillna(0.0) * mult_used
        prof_col = _find_col(out.columns.tolist(), PROFIT_COLS)
        if prof_col:
            out[prof_col] = out[ret_cols[0]] - out[stake_col]
    else:
        for pc in [c for c in PROFIT_COLS if c in out.columns]:
            out[pc] = pd.to_numeric(out[pc], errors="coerce").fillna(0.0) * mult_used
    return out


def _parse_plan(rolling_index: dict, window_name: Optional[str]) -> dict:
    plans = rolling_index.get("plans") or []
    if not plans:
        raise ValueError("rolling_index has no plans")
    if window_name:
        for plan in plans:
            if plan.get("window_name") == window_name:
                return plan
        raise ValueError(f"window_name not found: {window_name}")
    return plans[0]


def _parse_cmd_flags(plan: dict) -> tuple[bool, Optional[float]]:
    cmd = plan.get("cmd") or []
    estimate = "--estimate-closing-mult" in cmd
    closing_q = None
    if "--closing-mult-quantile" in cmd:
        idx = cmd.index("--closing-mult-quantile")
        if idx + 1 < len(cmd):
            try:
                closing_q = float(cmd[idx + 1])
            except Exception:
                closing_q = None
    return estimate, closing_q


def _run_holdout(
    *,
    cfg_path: Path,
    plan: dict,
    out_dir: Path,
    estimate_closing: bool,
    closing_q: Optional[float],
    initial_bankroll: Optional[float],
    audit_log: Path,
) -> None:
    run_holdout = Path(__file__).resolve().parents[1] / "scripts" / "run_holdout.py"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(Path.cwd() / "py64_analysis" / ".venv" / "Scripts" / "python.exe"),
        str(run_holdout),
        "--train-start",
        plan["train_start"],
        "--train-end",
        plan["train_end"],
        "--test-start",
        plan["test_start"],
        "--test-end",
        plan["test_end"],
        "--name",
        plan["window_name"],
        "--out-dir",
        str(out_dir),
    ]
    if plan.get("valid_start") and plan.get("valid_end"):
        cmd += ["--valid-start", plan["valid_start"], "--valid-end", plan["valid_end"]]
    if estimate_closing:
        cmd.append("--estimate-closing-mult")
    if closing_q is not None:
        cmd += ["--closing-mult-quantile", str(closing_q)]
    if initial_bankroll is not None:
        cmd += ["--initial-bankroll", str(initial_bankroll)]

    env = os.environ.copy()
    env["KEIBA_CONFIG_PATH"] = str(cfg_path)
    env["PYTHONHASHSEED"] = "0"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    proc = subprocess.run(cmd, cwd=Path.cwd(), env=env, capture_output=True, text=True)
    with audit_log.open("a", encoding="utf-8") as fh:
        fh.write(proc.stdout)
        fh.write("\n")
        fh.write(proc.stderr)
        fh.write("\n")
    if proc.returncode != 0:
        raise RuntimeError(f"run_holdout failed: {cfg_path} -> {proc.returncode}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--variant-config", required=True)
    ap.add_argument("--rolling-index", required=True)
    ap.add_argument("--window-name", default=None)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_dir = out_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_log = audit_dir / "audit_stdout.txt"
    audit_log.write_text("", encoding="utf-8")

    rolling_index = json.loads(Path(args.rolling_index).read_text(encoding="utf-8"))
    plan = _parse_plan(rolling_index, args.window_name)
    estimate_closing, closing_q = _parse_cmd_flags(plan)
    initial_bankroll = rolling_index.get("args", {}).get("initial_bankroll")

    base_run = audit_dir / "engine_runs" / "base"
    var_run = audit_dir / "engine_runs" / "variant"
    base_run.mkdir(parents=True, exist_ok=True)
    var_run.mkdir(parents=True, exist_ok=True)

    _run_holdout(
        cfg_path=Path(args.base_config),
        plan=plan,
        out_dir=base_run,
        estimate_closing=estimate_closing,
        closing_q=closing_q,
        initial_bankroll=initial_bankroll,
        audit_log=audit_log,
    )
    _run_holdout(
        cfg_path=Path(args.variant_config),
        plan=plan,
        out_dir=var_run,
        estimate_closing=estimate_closing,
        closing_q=closing_q,
        initial_bankroll=initial_bankroll,
        audit_log=audit_log,
    )

    base_bets_path = base_run / "bets.csv"
    var_bets_path = var_run / "bets.csv"
    if not base_bets_path.exists() or not var_bets_path.exists():
        raise FileNotFoundError("missing bets.csv in base or variant run")

    base_df = pd.read_csv(base_bets_path)
    var_df = pd.read_csv(var_bets_path)

    stake_col = _find_col(base_df.columns.tolist(), STAKE_COLS)
    odds_col = _find_col(base_df.columns.tolist(), ODDS_COLS)
    key_cols = _detect_key_cols(base_df.columns.tolist())
    if stake_col is None or odds_col is None or key_cols is None:
        raise RuntimeError("required columns missing in base bets.csv")

    cfg_used = _read_yaml(base_run / "config_used.yaml")
    min_yen = int(((cfg_used.get("betting") or {}).get("stake") or {}).get("min_yen", 100))

    var_cfg = _read_yaml(Path(args.variant_config))
    damp_cfg = ((var_cfg.get("betting") or {}).get("stake_odds_damp") or {})
    ref_odds = float(damp_cfg.get("ref_odds", 0.0) or 0.0)
    power = float(damp_cfg.get("power", 1.0) or 1.0)
    min_mult = float(damp_cfg.get("min_mult", 0.0) or 0.0)

    replay_df = _apply_stake_damp(
        base_df,
        stake_col=stake_col,
        odds_col=odds_col,
        ref_odds=ref_odds,
        power=power,
        min_mult=min_mult,
        min_yen=min_yen,
    )
    replay_path = audit_dir / "replay_expected_bets.csv"
    replay_df.to_csv(replay_path, index=False, encoding="utf-8")

    extra_cols = []
    if _find_col(base_df.columns.tolist(), RETURN_COLS):
        extra_cols.append(_find_col(base_df.columns.tolist(), RETURN_COLS))
    if _find_col(base_df.columns.tolist(), PROFIT_COLS):
        extra_cols.append(_find_col(base_df.columns.tolist(), PROFIT_COLS))
    extra_cols = [c for c in extra_cols if c]

    replay_hash = _hash_rows(replay_df, key_cols, stake_col, odds_col, extra_cols)
    var_hash = _hash_rows(var_df, key_cols, stake_col, odds_col, extra_cols)
    replay_keys = set(replay_df[key_cols].astype(str).agg("|".join, axis=1))
    var_keys = set(var_df[key_cols].astype(str).agg("|".join, axis=1))

    rowcount_match = len(replay_df) == len(var_df)
    key_match = replay_keys == var_keys
    hash_match = replay_hash == var_hash

    diff_path = audit_dir / "diff.csv"
    if not (rowcount_match and key_match and hash_match):
        left = replay_df[key_cols + [stake_col]].copy()
        right = var_df[key_cols + [stake_col]].copy()
        left["__side__"] = "replay"
        right["__side__"] = "engine"
        merged = pd.merge(left, right, on=key_cols, how="outer", suffixes=("_replay", "_engine"), indicator=True)
        mism = merged[(merged["_merge"] != "both") | (merged[f"{stake_col}_replay"] != merged[f"{stake_col}_engine"])]
        mism.head(500).to_csv(diff_path, index=False, encoding="utf-8")

    # no-op check (ref_odds=0)
    noop_df = _apply_stake_damp(
        base_df,
        stake_col=stake_col,
        odds_col=odds_col,
        ref_odds=0.0,
        power=power,
        min_mult=0.0,
        min_yen=min_yen,
    )
    noop_hash = _hash_rows(noop_df, key_cols, stake_col, odds_col, extra_cols)
    noop_pass = bool(len(noop_df) == len(base_df) and noop_hash == _hash_rows(base_df, key_cols, stake_col, odds_col, extra_cols))

    audit_result = {
        "window_name": plan["window_name"],
        "rowcount_match": rowcount_match,
        "key_match": key_match,
        "hash_match": hash_match,
        "replay_hash": replay_hash,
        "engine_hash": var_hash,
        "no_op_pass": noop_pass,
        "ref_odds": ref_odds,
        "power": power,
        "min_mult": min_mult,
        "base_run_dir": str(base_run),
        "variant_run_dir": str(var_run),
    }
    (audit_dir / "audit_result.json").write_text(json.dumps(audit_result, indent=2), encoding="utf-8")

    # artifact samples
    samples = out_dir / "artifact_samples"
    samples.mkdir(parents=True, exist_ok=True)
    for label, run_dir in [("base", base_run), ("variant", var_run)]:
        dest = samples / label
        dest.mkdir(parents=True, exist_ok=True)
        for name in ["config_used.yaml", "summary.json"]:
            src = run_dir / name
            if src.exists():
                (dest / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        bets_path = run_dir / "bets.csv"
        if bets_path.exists():
            df = pd.read_csv(bets_path)
            df.head(100).to_csv(dest / "bets_head100.csv", index=False, encoding="utf-8")

    print(
        f"[parity] stake_odds_damp_replay_vs_engine_pass={str(rowcount_match and key_match and hash_match).lower()} | "
        f"setting=ref_odds={ref_odds},power={power},min_mult={min_mult} | window={plan['window_name']}"
    )
    print(f"[noop] stake_odds_damp_noop_pass={str(noop_pass).lower()}")
    return 0 if (rowcount_match and key_match and hash_match and noop_pass) else 1


if __name__ == "__main__":
    raise SystemExit(main())
