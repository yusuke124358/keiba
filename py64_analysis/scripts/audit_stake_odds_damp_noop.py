"""Audit stake odds damp no-op and bet-set invariance on a single window."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


STAKE_COLS = ["stake", "stake_yen", "stake_amount", "bet_amount", "stake_raw"]
RETURN_COLS = ["return", "return_yen", "payout", "payout_yen"]
PROFIT_COLS = ["profit", "net_profit", "pnl"]
ODDS_COLS = ["odds_at_buy", "odds", "buy_odds", "win_odds"]
KEY_COL_CANDIDATES = [
    ["race_id", "horse_no", "asof_time"],
    ["race_id", "horse_id", "asof_time"],
]


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


def _hash_rows(df: pd.DataFrame, key_cols: list[str], stake_col: str, odds_col: str) -> str:
    key = df[key_cols].astype(str).agg(lambda r: "|".join(r.values), axis=1)
    stake = pd.to_numeric(df[stake_col], errors="coerce").fillna(0.0).astype(float)
    odds = pd.to_numeric(df[odds_col], errors="coerce").fillna(0.0).astype(float)
    joined = (key + "|" + stake.map(lambda v: f"{v:.1f}") + "|" + odds.map(lambda v: f"{v:.6f}")).tolist()
    return hashlib.sha1("\n".join(sorted(joined)).encode("utf-8")).hexdigest()


def _apply_stake_damp(
    df: pd.DataFrame,
    stake_col: str,
    odds_col: str,
    ref_odds: float,
    min_mult: float,
    min_yen: int,
    power: float,
) -> pd.DataFrame:
    df = df.copy()
    stake = pd.to_numeric(df[stake_col], errors="coerce").fillna(0.0)
    odds = pd.to_numeric(df[odds_col], errors="coerce")
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
    ratio = pd.Series([np.nan] * len(df), index=df.index)
    mult_raw = pd.Series([1.0] * len(df), index=df.index)
    mult = mult_raw.copy()
    stake_after = stake.copy()
    if ref_odds > 0:
        valid = (odds > 0) & np.isfinite(odds)
        ratio = odds / float(ref_odds)
        ratio = ratio.where(valid, np.nan)
        low_mask = ratio < 1.0
        raw = pd.Series(np.power(ratio, power), index=df.index)
        raw = raw.where(np.isfinite(raw) & (raw > 0), 0.0)
        raw = raw.where(low_mask, 1.0)
        mult_raw = raw.where(valid, 1.0)
        mult = np.maximum(min_mult, mult_raw)
        mult = np.minimum(1.0, mult)
        mult = pd.Series(mult, index=df.index)
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
    df[stake_col] = stake_after
    df["__stake_mult__"] = mult_used
    ret_cols = [c for c in RETURN_COLS if c in df.columns]
    if ret_cols:
        for rc in ret_cols:
            df[rc] = pd.to_numeric(df[rc], errors="coerce").fillna(0.0) * mult_used
        prof_col = _find_col(df.columns.tolist(), PROFIT_COLS)
        if prof_col:
            df[prof_col] = df[ret_cols[0]] - df[stake_col]
    else:
        for pc in [c for c in PROFIT_COLS if c in df.columns]:
            df[pc] = pd.to_numeric(df[pc], errors="coerce").fillna(0.0) * mult_used
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_run_dir", required=True)
    ap.add_argument("--ref_odds", type=float, default=6.0)
    ap.add_argument("--min_mult", type=float, default=0.0)
    ap.add_argument("--power", type=float, default=1.0)
    ap.add_argument("--window")
    args = ap.parse_args()

    base_dir = Path(args.base_run_dir)
    windows = [Path(args.window)] if args.window else sorted(base_dir.glob("w*/"))
    target = None
    for w in windows:
        bets_path = w / "bets.csv"
        if bets_path.exists():
            df = pd.read_csv(bets_path)
            if not df.empty:
                target = w
                break
    if target is None:
        print("[noop] stake_odds_damp_noop_pass=false | checked_window=none | ref_odds=6.0")
        return 1

    df = pd.read_csv(target / "bets.csv")
    stake_col = _find_col(df.columns.tolist(), STAKE_COLS)
    odds_col = _find_col(df.columns.tolist(), ODDS_COLS)
    key_cols = _detect_key_cols(df.columns.tolist())
    if stake_col is None or odds_col is None or key_cols is None:
        print(f"[noop] stake_odds_damp_noop_pass=false | checked_window={target.name} | ref_odds=6.0")
        return 1

    min_yen = 100
    cfg_path = target / "config_used.yaml"
    if cfg_path.exists():
        try:
            import yaml

            data = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            stake_cfg = (data.get("betting") or {}).get("stake") or {}
            min_yen = int(stake_cfg.get("min_yen", 100))
        except Exception:
            min_yen = 100

    power = float(args.power)
    if not np.isfinite(power) or power <= 0:
        power = 1.0
    noop_df = _apply_stake_damp(df, stake_col, odds_col, ref_odds=0.0, min_mult=0.0, min_yen=min_yen, power=power)
    rowcount_match = len(noop_df) == len(df)
    hash_match = _hash_rows(noop_df, key_cols, stake_col, odds_col) == _hash_rows(df, key_cols, stake_col, odds_col)

    damp_df = _apply_stake_damp(
        df,
        stake_col,
        odds_col,
        ref_odds=args.ref_odds,
        min_mult=args.min_mult,
        min_yen=min_yen,
        power=power,
    )
    rowcount_match_damp = len(damp_df) == len(df)
    key_match = set(damp_df[key_cols].astype(str).agg("|".join, axis=1)) == set(df[key_cols].astype(str).agg("|".join, axis=1))

    sample_ok = True
    ret_col = _find_col(df.columns.tolist(), RETURN_COLS)
    prof_col = _find_col(df.columns.tolist(), PROFIT_COLS)
    if ret_col or prof_col:
        base_ret = pd.to_numeric(df[ret_col], errors="coerce") if ret_col else None
        base_prof = pd.to_numeric(df[prof_col], errors="coerce") if prof_col else None
        mult = damp_df["__stake_mult__"]
        idx = mult.index[:10]
        if ret_col:
            expect = base_ret.loc[idx] * mult.loc[idx]
            got = pd.to_numeric(damp_df.loc[idx, ret_col], errors="coerce") if ret_col in damp_df.columns else None
            if got is not None and not np.allclose(expect.fillna(0), got.fillna(0), atol=1e-6):
                sample_ok = False
        if prof_col:
            expect = base_prof.loc[idx] * mult.loc[idx]
            got = pd.to_numeric(damp_df.loc[idx, prof_col], errors="coerce") if prof_col in damp_df.columns else None
            if got is not None and not np.allclose(expect.fillna(0), got.fillna(0), atol=1e-6):
                sample_ok = False

    noop_pass = bool(rowcount_match and hash_match and rowcount_match_damp and key_match and sample_ok)
    print(
        f"[noop] stake_odds_damp_noop_pass={str(noop_pass).lower()} | checked_window={target.name} | ref_odds={args.ref_odds} | power={power}"
    )
    return 0 if noop_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
