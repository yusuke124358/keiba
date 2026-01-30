"""Audit odds_floor no-op and replay parity for a single window."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Optional

import pandas as pd


KEY_COL_CANDIDATES = [
    ["race_id", "horse_no", "asof_time"],
    ["race_id", "horse_id", "asof_time"],
]
STAKE_COLS = ["stake", "stake_yen", "stake_amount", "bet_amount", "stake_raw"]
ODDS_COLS = ["odds_at_buy", "odds", "buy_odds", "win_odds"]


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


def _hash_rows(df: pd.DataFrame, key_cols: list[str], stake_col: str, odds_col: str) -> str:
    key = df[key_cols].astype(str).agg(lambda r: "|".join(r.values), axis=1)
    stake = pd.to_numeric(df[stake_col], errors="coerce").fillna(0.0).astype(float)
    odds = pd.to_numeric(df[odds_col], errors="coerce").fillna(0.0).astype(float)
    joined = (key + "|" + stake.map(lambda v: f"{v:.1f}") + "|" + odds.map(lambda v: f"{v:.6f}")).tolist()
    digest = hashlib.sha1("\n".join(sorted(joined)).encode("utf-8")).hexdigest()
    return digest


def _load_bets(run_dir: Path, window: str) -> pd.DataFrame:
    bets_path = run_dir / window / "bets.csv"
    if not bets_path.exists():
        raise FileNotFoundError(f"bets.csv not found: {bets_path}")
    return pd.read_csv(bets_path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-run-dir", required=True)
    ap.add_argument("--variant-run-dir", required=True)
    ap.add_argument("--min-odds", type=float, required=True)
    ap.add_argument("--window", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    base_dir = Path(args.base_run_dir)
    var_dir = Path(args.variant_run_dir)
    window = args.window
    if window is None:
        windows = sorted([p.name for p in base_dir.glob("w*/")])
        if not windows:
            raise SystemExit(f"No windows found under {base_dir}")
        window = windows[0]

    bdf = _load_bets(base_dir, window)
    vdf = _load_bets(var_dir, window)

    key_cols = _detect_key_cols(list(bdf.columns))
    if key_cols is None or any(c not in vdf.columns for c in key_cols):
        raise SystemExit("Could not detect compatible key columns for base/variant bets.csv.")

    stake_col = _find_col(list(bdf.columns), STAKE_COLS)
    odds_col = _find_col(list(bdf.columns), ODDS_COLS)
    if stake_col is None or odds_col is None:
        raise SystemExit("Could not detect stake/odds columns for audit.")

    bdf["__key__"] = bdf[key_cols].astype(str).agg(lambda r: "|".join(r.values), axis=1)
    vdf["__key__"] = vdf[key_cols].astype(str).agg(lambda r: "|".join(r.values), axis=1)

    # No-op check (base vs variant direct)
    base_hash = _hash_rows(bdf, key_cols, stake_col, odds_col)
    var_hash = _hash_rows(vdf, key_cols, stake_col, odds_col)
    noop_pass = (len(bdf) == len(vdf)) and (base_hash == var_hash)

    # Replay parity check (variant vs base filtered by min_odds)
    min_odds = float(args.min_odds)
    if min_odds > 0:
        odds_vals = pd.to_numeric(bdf[odds_col], errors="coerce").fillna(0.0)
        filtered = bdf[odds_vals >= min_odds].copy()
    else:
        filtered = bdf.copy()

    base_f_hash = _hash_rows(filtered, key_cols, stake_col, odds_col) if not filtered.empty else None
    var_f_hash = _hash_rows(vdf, key_cols, stake_col, odds_col) if not vdf.empty else None
    parity_pass = (len(filtered) == len(vdf)) and (base_f_hash == var_f_hash)

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = {
            "window": window,
            "min_odds": min_odds,
            "base_rows": int(len(bdf)),
            "variant_rows": int(len(vdf)),
            "filtered_base_rows": int(len(filtered)),
            "noop_pass": bool(noop_pass),
            "parity_pass": bool(parity_pass),
        }
        (out_dir / "odds_floor_parity_audit.json").write_text(
            pd.Series(out).to_json(indent=2, force_ascii=False), encoding="utf-8"
        )

    print(f"[noop] odds_floor_noop_pass={str(noop_pass).lower()}")
    print(f"[parity] odds_floor_engine_matches_replay={str(parity_pass).lower()} | min_odds={min_odds} | window={window}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
