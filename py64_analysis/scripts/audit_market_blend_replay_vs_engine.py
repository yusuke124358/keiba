"""Audit market-blend replay vs engine outputs on a single window."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml


KEY_COL_CANDIDATES = [
    ["race_id", "horse_no", "asof_time"],
    ["race_id", "horse_id", "asof_time"],
    ["race_id", "horse_no"],
    ["race_id", "horse_id"],
]
STAKE_COLS = ["stake", "stake_yen", "stake_amount", "bet_amount", "stake_raw"]
ODDS_COLS = ["odds_at_buy", "odds", "buy_odds", "win_odds"]

TOL_FLOAT = 1e-6


def _ensure_import_path() -> None:
    project_root = Path(__file__).resolve().parents[2]
    src = project_root / "py64_analysis" / "src"
    if src.exists():
        sys.path.insert(0, str(src))


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _detect_key_cols(cols: list[str]) -> Optional[list[str]]:
    for cand in KEY_COL_CANDIDATES:
        if all(c in cols for c in cand):
            return cand
    return None


def _find_col(cols: list[str], candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in cols:
            return c
    return None


def _parse_plan(rolling_index: dict, window_name: Optional[str]) -> dict:
    plans = rolling_index.get("plans") or []
    if not plans:
        raise ValueError("rolling_index has no plans")
    if window_name:
        for plan in plans:
            if plan.get("window_name") == window_name:
                return plan
        raise ValueError(f"window_name not found: {window_name}")
    # prefer specific window if present
    for plan in plans:
        if plan.get("window_name") == "w001_20250101_20250301":
            return plan
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
    project_root = Path(__file__).resolve().parents[2]
    run_holdout = project_root / "py64_analysis" / "scripts" / "run_holdout.py"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(project_root / "py64_analysis" / ".venv" / "Scripts" / "python.exe"),
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
    proc = subprocess.run(cmd, cwd=str(project_root), env=env, capture_output=True, text=True)
    with audit_log.open("a", encoding="utf-8") as fh:
        fh.write(proc.stdout)
        fh.write("\n")
        fh.write(proc.stderr)
        fh.write("\n")
    if proc.returncode != 0:
        raise RuntimeError(f"run_holdout failed: {cfg_path} -> {proc.returncode}")


def _to_float(val: object) -> Optional[float]:
    if val is None:
        return None
    try:
        out = float(val)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _read_race_ids(path: Path) -> list[str]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        val = line.strip()
        if val:
            rows.append(val)
    return rows


def _build_fullfield_df(run_dir: Path) -> pd.DataFrame:
    _ensure_import_path()
    from keiba.db.loader import get_session
    from keiba.backtest.engine import BacktestEngine
    from keiba.modeling.train import WinProbabilityModel

    cfg_path = run_dir / "config_used.yaml"
    if cfg_path.exists():
        os.environ["KEIBA_CONFIG_PATH"] = str(cfg_path)
    else:
        os.environ.pop("KEIBA_CONFIG_PATH", None)

    session = get_session()
    model_path = run_dir / "artifacts" / "model.pkl"
    if not model_path.exists():
        model_path = run_dir / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"model.pkl not found under {run_dir}")
    model = WinProbabilityModel.load(model_path)
    engine = BacktestEngine(session)

    race_ids = _read_race_ids(run_dir / "race_ids_test.txt")
    if not race_ids:
        raise FileNotFoundError(f"race_ids_test.txt missing: {run_dir}")

    rows: list[dict] = []
    for race_id in race_ids:
        buy_time = engine._get_buy_time(race_id)
        if buy_time is None:
            continue
        preds = engine._predict_race(race_id, buy_time, model)
        for pred in preds:
            odds_val = _to_float(pred.get("odds"))
            rows.append(
                {
                    "race_id": race_id,
                    "horse_no": int(pred.get("horse_no") or 0),
                    "asof_time": buy_time.isoformat(sep=" "),
                    "odds_at_buy": odds_val,
                    "p_hat_raw": _to_float(pred.get("p_hat")),
                    "p_model": _to_float(pred.get("p_model")),
                    "p_mkt": _to_float(pred.get("p_mkt")),
                    "has_ts_odds": bool(pred.get("has_ts_odds", True)),
                    "log_odds_std_60m": _to_float(pred.get("log_odds_std_60m")),
                    "snap_age_min": _to_float(pred.get("snap_age_min")),
                }
            )
    session.close()
    return pd.DataFrame(rows)


def _compute_replay_bets(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    _ensure_import_path()
    from keiba.betting.market_blend import logit_blend_prob, odds_band, parse_exclude_odds_band
    from keiba.betting.sizing import calculate_stake
    from keiba.db.loader import get_session
    from sqlalchemy import text

    if df.empty:
        return df

    betting = cfg.get("betting", {}) if isinstance(cfg, dict) else {}
    backtest = cfg.get("backtest", {}) if isinstance(cfg, dict) else {}

    if not bool(betting.get("enable_market_blend", False)):
        raise RuntimeError("enable_market_blend is false; replay audit expects market_blend enabled.")
    if str(betting.get("market_prob_method", "p_mkt_col")).lower() != "p_mkt_col":
        raise RuntimeError("market_prob_method must be p_mkt_col for replay audit.")

    # guard: replay assumes these are disabled
    if (betting.get("odds_band_bias") or {}).get("enabled"):
        raise RuntimeError("odds_band_bias enabled; replay does not support this audit.")
    if betting.get("overlay_shrink_alpha") is not None:
        raise RuntimeError("overlay_shrink_alpha enabled; replay does not support this audit.")
    if (betting.get("race_cost_filter") or {}).get("enabled"):
        raise RuntimeError("race_cost_filter enabled; replay does not support this audit.")
    if (betting.get("odds_dynamics_filter") or {}).get("enabled"):
        raise RuntimeError("odds_dynamics_filter enabled; replay does not support this audit.")
    if (betting.get("odds_dyn_ev_margin") or {}).get("enabled"):
        raise RuntimeError("odds_dyn_ev_margin enabled; replay does not support this audit.")
    if betting.get("ev_cap_quantile") is not None or betting.get("overlay_abs_cap_quantile") is not None:
        raise RuntimeError("ev/overlay caps enabled; replay does not support this audit.")

    t_ev = float(betting.get("t_ev", betting.get("ev_margin", 0.0)) or 0.0)
    odds_cap = betting.get("odds_cap")
    try:
        odds_cap_val = float(odds_cap) if odds_cap is not None else None
    except Exception:
        odds_cap_val = None
    exclude_bands = parse_exclude_odds_band(betting.get("exclude_odds_band"))
    blend_w = float(betting.get("market_blend_w", 1.0))
    if not math.isfinite(blend_w):
        blend_w = 1.0

    min_odds = betting.get("min_odds")
    max_odds = betting.get("max_odds")
    min_buy_odds = betting.get("min_buy_odds")
    max_buy_odds = betting.get("max_buy_odds")

    closing_mult = float(betting.get("closing_odds_multiplier", 1.0) or 1.0)
    slippage = backtest.get("slippage", {}) if isinstance(backtest, dict) else {}
    slippage_enabled = bool(slippage.get("enabled_if_no_ts_odds", False))
    slippage_mult = float(slippage.get("odds_multiplier", 1.0) or 1.0)

    stake_cfg = betting.get("stake", {}) if isinstance(betting, dict) else {}
    stake_enabled = bool(stake_cfg.get("enabled", False))
    min_yen = int(stake_cfg.get("min_yen", 100) or 100)
    bankroll_init = float(betting.get("bankroll_yen", 0) or 0)
    max_pct = float((betting.get("caps") or {}).get("per_race_pct", 0.0) or 0.0)
    per_day_pct = (betting.get("caps") or {}).get("per_day_pct", None)
    per_day_cap = float(bankroll_init) * float(per_day_pct) if per_day_pct is not None else None
    max_daily_loss = float(betting.get("stop", {}).get("max_daily_loss_pct", 0.0) or 0.0) * float(bankroll_init)
    max_bets = int(betting.get("max_bets_per_race", 1) or 1)

    # results map
    race_ids = sorted(set(df["race_id"].astype(str).tolist()))
    result_map: dict[tuple[str, int], tuple[int, Optional[float]]] = {}
    session = get_session()
    for i in range(0, len(race_ids), 500):
        chunk = race_ids[i : i + 500]
        rows = session.execute(
            text(
                """
                SELECT race_id, horse_no, finish_pos, odds
                FROM fact_result
                WHERE race_id = ANY(:race_ids)
                """
            ),
            {"race_ids": chunk},
        ).fetchall()
        for r in rows:
            rid = str(r[0])
            hno = int(r[1])
            fin = int(r[2]) if r[2] is not None else 99
            odds_final = float(r[3]) if r[3] is not None else None
            result_map[(rid, hno)] = (fin, odds_final)
    session.close()

    def _profit_for(race_id: str, horse_no: int, stake: float, odds_at_buy: float) -> float:
        fin, odds_final = result_map.get((race_id, horse_no), (99, None))
        odds_use = odds_at_buy if odds_final is None else float(odds_final)
        if fin == 1:
            payout = stake * odds_use
        else:
            payout = 0.0
        return float(payout - stake)

    out = df.copy()
    out = out.dropna(subset=["odds_at_buy", "p_model", "p_mkt", "p_hat_raw"])
    out = out[out["p_hat_raw"] > 0]
    out["race_id"] = out["race_id"].astype(str)
    out["asof_dt"] = pd.to_datetime(out["asof_time"], errors="coerce")

    # apply odds-based filters
    if min_buy_odds is not None:
        out = out[out["odds_at_buy"] >= float(min_buy_odds)]
    if max_buy_odds is not None:
        out = out[out["odds_at_buy"] <= float(max_buy_odds)]
    if min_odds is not None:
        out = out[out["odds_at_buy"] >= float(min_odds)]
    if max_odds is not None:
        out = out[out["odds_at_buy"] <= float(max_odds)]

    if out.empty:
        return out

    out = out.sort_values(["race_id", "horse_no"])

    bankroll_decision = float(bankroll_init)
    current_date = None
    daily_stake = 0.0
    daily_loss = 0.0

    selected_rows = []
    race_ids_sorted = sorted(out["race_id"].astype(str).unique().tolist())
    for race_id in race_ids_sorted:
        race_df = out[out["race_id"] == race_id]
        if race_df.empty:
            continue
        asof_dt = race_df["asof_dt"].iloc[0]
        if pd.isna(asof_dt):
            continue
        race_date = None
        if isinstance(race_id, str) and len(race_id) >= 8 and race_id[:8].isdigit():
            race_date = race_id[:8]
        if race_date is None:
            race_date = asof_dt.strftime("%Y%m%d")
        if current_date != race_date:
            current_date = race_date
            daily_stake = 0.0
            daily_loss = 0.0
        if max_daily_loss > 0 and daily_loss >= max_daily_loss:
            continue

        remaining_budget = None
        if per_day_cap is not None:
            remaining_budget = max(0.0, per_day_cap - daily_stake)

        cand_rows = []
        for row in race_df.itertuples(index=False):
            odds_at_buy = float(row.odds_at_buy)
            p_model = float(row.p_model)
            p_mkt = float(row.p_mkt)
            if odds_at_buy <= 0 or p_model <= 0 or p_mkt <= 0:
                continue

            if slippage_enabled and not bool(row.has_ts_odds):
                odds_at_buy = odds_at_buy * slippage_mult

            odds_effective = odds_at_buy * closing_mult if closing_mult > 0 else odds_at_buy

            p_blend = logit_blend_prob(p_model, p_mkt, blend_w)
            ev_blend = float(p_blend) * odds_effective - 1.0
            odds_band_blend = odds_band(odds_at_buy)

            if odds_cap_val is not None and odds_at_buy > odds_cap_val:
                continue
            if exclude_bands and odds_band_blend in exclude_bands:
                continue
            if ev_blend < t_ev:
                continue

            stake = calculate_stake(
                p_hat=float(p_blend),
                odds=float(odds_effective),
                bankroll=bankroll_decision,
                method=betting.get("sizing", {}).get("method", "fractional_kelly"),
                fraction=float((betting.get("sizing", {}) or {}).get("fraction", 0.2) or 0.2),
                max_pct=1.0 if stake_enabled else max_pct,
                min_stake=min_yen,
            )
            if stake < min_yen:
                continue

            cand_rows.append(
                {
                    "race_id": race_id,
                    "horse_no": int(row.horse_no),
                    "asof_time": row.asof_time,
                    "odds_at_buy": odds_at_buy,
                    "odds_effective": odds_effective,
                    "p_blend": float(p_blend),
                    "ev_blend": ev_blend,
                    "stake": float(stake),
                }
            )

        if not cand_rows:
            continue
        cand_rows = sorted(cand_rows, key=lambda r: r["ev_blend"], reverse=True)[:max_bets]

        for cand in cand_rows:
            if remaining_budget is not None and cand["stake"] > remaining_budget:
                continue
            selected_rows.append(cand)
            daily_stake += cand["stake"]
            if remaining_budget is not None:
                remaining_budget = max(0.0, per_day_cap - daily_stake)
            profit_before = _profit_for(cand["race_id"], cand["horse_no"], cand["stake"], cand["odds_at_buy"])
            bankroll_decision += profit_before
            if profit_before < 0:
                daily_loss += abs(profit_before)

    return pd.DataFrame(selected_rows)


def _compare_frames(replay: pd.DataFrame, engine: pd.DataFrame, key_cols: list[str], out_dir: Path) -> dict:
    diff_path = out_dir / "diff.csv"
    if replay.empty or engine.empty:
        return {
            "rowcount_match": len(replay) == len(engine),
            "key_match": False,
            "stake_match": False,
            "derived_match": False,
            "diff_path": str(diff_path),
        }

    for c in key_cols:
        if c in replay.columns:
            replay[c] = replay[c].astype(str)
        if c in engine.columns:
            engine[c] = engine[c].astype(str)

    merged = pd.merge(
        replay,
        engine,
        on=key_cols,
        how="outer",
        suffixes=("_replay", "_engine"),
        indicator=True,
    )

    rowcount_match = len(replay) == len(engine)
    key_match = merged["_merge"].eq("both").all()

    stake_diff = (merged["stake_replay"] - merged["stake_engine"]).abs()
    stake_match = stake_diff.fillna(0).le(0).all()

    derived_cols = ["odds_at_buy", "odds_effective", "p_blend", "ev_blend"]
    derived_match = True
    for col in derived_cols:
        rcol = f"{col}_replay"
        ecol = f"{col}_engine"
        if rcol not in merged.columns or ecol not in merged.columns:
            continue
        diff = (merged[rcol] - merged[ecol]).abs()
        if diff.fillna(0).gt(TOL_FLOAT).any():
            derived_match = False
            break

    if not (rowcount_match and key_match and stake_match and derived_match):
        mism = merged.copy()
        for col in ["stake", "odds_at_buy", "odds_effective", "p_blend", "ev_blend"]:
            rcol = f"{col}_replay"
            ecol = f"{col}_engine"
            if rcol in mism.columns and ecol in mism.columns:
                mism[f"diff_{col}"] = (mism[rcol] - mism[ecol]).abs()
        mism.head(500).to_csv(diff_path, index=False, encoding="utf-8")

    return {
        "rowcount_match": bool(rowcount_match),
        "key_match": bool(key_match),
        "stake_match": bool(stake_match),
        "derived_match": bool(derived_match),
        "diff_path": str(diff_path),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--rolling-index", required=True)
    ap.add_argument("--window-name", default=None)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audit_log = out_dir / "audit_stdout.txt"
    audit_log.write_text("", encoding="utf-8")

    rolling_index = json.loads(Path(args.rolling_index).read_text(encoding="utf-8"))
    plan = _parse_plan(rolling_index, args.window_name)
    estimate_closing, closing_q = _parse_cmd_flags(plan)
    initial_bankroll = rolling_index.get("args", {}).get("initial_bankroll")

    engine_run = out_dir / "engine_run"
    _run_holdout(
        cfg_path=Path(args.config),
        plan=plan,
        out_dir=engine_run,
        estimate_closing=estimate_closing,
        closing_q=closing_q,
        initial_bankroll=initial_bankroll,
        audit_log=audit_log,
    )

    engine_bets_path = engine_run / "bets.csv"
    if not engine_bets_path.exists():
        raise FileNotFoundError(f"bets.csv missing: {engine_bets_path}")

    engine_df = pd.read_csv(engine_bets_path)
    key_cols = _detect_key_cols(list(engine_df.columns))
    stake_col = _find_col(list(engine_df.columns), STAKE_COLS)
    if key_cols is None or stake_col is None:
        raise RuntimeError("required key/stake columns missing in engine bets.csv")

    cfg_used = _read_yaml(engine_run / "config_used.yaml")
    replay_fullfield = _build_fullfield_df(engine_run)
    replay_bets = _compute_replay_bets(replay_fullfield, cfg_used)

    replay_path = out_dir / "replay_expected_bets.csv"
    replay_bets.to_csv(replay_path, index=False, encoding="utf-8")

    # normalize to stake column name for compare
    replay_cmp = replay_bets.copy()
    replay_cmp["stake"] = pd.to_numeric(replay_cmp.get("stake"), errors="coerce").fillna(0.0)
    engine_cmp = engine_df.copy()
    if stake_col != "stake":
        engine_cmp["stake"] = pd.to_numeric(engine_cmp[stake_col], errors="coerce").fillna(0.0)

    compare = _compare_frames(replay_cmp, engine_cmp, key_cols, out_dir)

    result = {
        "window_name": plan.get("window_name"),
        "rowcount_engine": int(len(engine_df)),
        "rowcount_replay": int(len(replay_bets)),
        "key_cols": key_cols,
        "stake_col": stake_col,
        "tolerance_float": TOL_FLOAT,
        **compare,
        "engine_run_dir": str(engine_run),
        "replay_path": str(replay_path),
    }
    (out_dir / "audit_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    parity_pass = bool(compare["rowcount_match"] and compare["key_match"] and compare["stake_match"] and compare["derived_match"])
    print(f"[parity] market_blend_replay_vs_engine_pass={str(parity_pass).lower()} | window={plan.get('window_name')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
