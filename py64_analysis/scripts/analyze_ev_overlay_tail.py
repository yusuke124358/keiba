"""
Analyze EV/overlay tail behavior from bets.csv (post-hoc, no reallocation).
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np


STAKE_COLS = ["stake", "stake_yen", "stake_amount", "total_stake", "stake_sum"]
PROFIT_COLS = ["profit", "net_profit", "pnl"]
ODDS_COLS = ["odds_at_buy", "odds", "buy_odds", "odds_buy"]
P_MKT_COLS = ["p_mkt", "market_prob"]
P_USED_COLS = ["p_cal", "p_blend", "p_hat", "p_adj", "p_hat_shrunk", "p_hat_raw"]
EV_COLS = ["ev", "ev_margin", "expected_value", "ev_adj"]


def _infer_w_idx_offset(run_dir_name: str) -> int:
    name = run_dir_name.lower()
    if "w013_022" in name or "w013-022" in name:
        return 12
    return 0


def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _to_num(s):
    return pd.to_numeric(s, errors="coerce")


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _clip_prob(p: float) -> float:
    return min(max(p, 1e-6), 1.0 - 1e-6)


def _assign_decile(values: pd.Series) -> pd.Series:
    x = values.copy()
    if x.dropna().empty:
        return pd.Series([np.nan] * len(values), index=values.index)
    try:
        dec = pd.qcut(x, 10, labels=False, duplicates="drop") + 1
        return dec.astype("float")
    except Exception:
        # fallback to rank-based bins
        ranks = x.rank(method="average")
        if ranks.dropna().empty:
            return pd.Series([np.nan] * len(values), index=values.index)
        dec = pd.qcut(ranks, 10, labels=False, duplicates="drop") + 1
        return dec.astype("float")


def _load_bets(run_dirs: list[Path]) -> pd.DataFrame:
    frames = []
    for run_dir in run_dirs:
        offset = _infer_w_idx_offset(run_dir.name)
        for bets_path in sorted(run_dir.rglob("w*/bets.csv")):
            window_name = bets_path.parent.name
            w_idx = None
            if window_name.startswith("w") and len(window_name) >= 4:
                try:
                    w_idx = int(window_name[1:4]) + int(offset)
                except Exception:
                    w_idx = None
            if w_idx is None:
                continue

            try:
                df = pd.read_csv(bets_path)
            except Exception as e:
                print(f"[warn] failed to read {bets_path}: {e}", file=sys.stderr)
                continue

            stake_col = _find_col(df, STAKE_COLS)
            profit_col = _find_col(df, PROFIT_COLS)
            odds_col = _find_col(df, ODDS_COLS)
            p_mkt_col = _find_col(df, P_MKT_COLS)
            p_used_col = _find_col(df, P_USED_COLS)
            ev_col = _find_col(df, EV_COLS)

            missing = [name for name, col in {
                "stake": stake_col,
                "profit": profit_col,
                "odds": odds_col,
                "p_mkt": p_mkt_col,
                "p_used": p_used_col,
            }.items() if col is None]
            if missing:
                print(f"[warn] missing columns in {bets_path}: {missing}", file=sys.stderr)
                continue

            tmp = pd.DataFrame({
                "stake": _to_num(df[stake_col]),
                "profit": _to_num(df[profit_col]),
                "odds": _to_num(df[odds_col]),
                "p_mkt": _to_num(df[p_mkt_col]),
                "p_used": _to_num(df[p_used_col]),
            })
            if ev_col is not None:
                tmp["ev"] = _to_num(df[ev_col])
            else:
                tmp["ev"] = tmp["p_used"] * tmp["odds"] - 1.0

            tmp = tmp.dropna(subset=["stake", "profit", "odds", "p_mkt", "p_used", "ev"])
            if tmp.empty:
                continue

            tmp["overlay_logit"] = tmp.apply(
                lambda r: _logit(_clip_prob(float(r["p_used"]))) - _logit(_clip_prob(float(r["p_mkt"]))),
                axis=1,
            )
            tmp["overlay_abs"] = tmp["overlay_logit"].abs()
            tmp["split"] = "design" if int(w_idx) <= 12 else "eval"
            frames.append(tmp)

    if not frames:
        return pd.DataFrame(columns=["stake", "profit", "odds", "p_mkt", "p_used", "ev", "overlay_logit", "overlay_abs", "split"])
    return pd.concat(frames, ignore_index=True)


def _summarize_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    out = df.groupby(group_cols, dropna=False).agg(
        stake=("stake", "sum"),
        profit=("profit", "sum"),
        n_bets=("stake", "count"),
    ).reset_index()
    out["roi"] = out.apply(lambda r: (r["profit"] / r["stake"]) if r["stake"] else float("nan"), axis=1)
    return out


def _quantile_sweep(df: pd.DataFrame, metric: str, quantiles: list[float]) -> pd.DataFrame:
    rows = []
    for split, g in df.groupby("split"):
        vals = g[metric].dropna()
        for q in quantiles:
            if vals.empty:
                rows.append({
                    "split": split,
                    "cap_q": q,
                    "threshold": float("nan"),
                    "stake": float("nan"),
                    "profit": float("nan"),
                    "roi": float("nan"),
                    "n_bets": 0,
                })
                continue
            thr = float(vals.quantile(q))
            kept = g[g[metric] <= thr]
            stake = float(kept["stake"].sum()) if not kept.empty else 0.0
            profit = float(kept["profit"].sum()) if not kept.empty else 0.0
            roi = (profit / stake) if stake else float("nan")
            rows.append({
                "split": split,
                "cap_q": q,
                "threshold": thr,
                "stake": stake,
                "profit": profit,
                "roi": roi,
                "n_bets": int(len(kept)),
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze EV/overlay tail behavior from bets.csv")
    ap.add_argument("--run-dirs", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    run_dirs = [Path(p) for p in args.run_dirs]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load_bets(run_dirs)
    if df.empty:
        raise SystemExit("No bets loaded; check run_dirs and required columns.")

    df["ev_decile"] = _assign_decile(df["ev"])
    df["overlay_decile"] = _assign_decile(df["overlay_abs"])

    ev_decile = _summarize_group(df.dropna(subset=["ev_decile"]), ["split", "ev_decile"])
    ev_decile = ev_decile.rename(columns={"ev_decile": "decile"})
    ev_decile.to_csv(out_dir / "ev_decile_table.csv", index=False, encoding="utf-8")

    ov_decile = _summarize_group(df.dropna(subset=["overlay_decile"]), ["split", "overlay_decile"])
    ov_decile = ov_decile.rename(columns={"overlay_decile": "decile"})
    ov_decile.to_csv(out_dir / "overlay_decile_table.csv", index=False, encoding="utf-8")

    qs = [0.95, 0.90, 0.85, 0.80]
    ev_sweep = _quantile_sweep(df, "ev", qs)
    ev_sweep.to_csv(out_dir / "ev_quantile_sweep.csv", index=False, encoding="utf-8")

    ov_sweep = _quantile_sweep(df, "overlay_abs", qs)
    ov_sweep.to_csv(out_dir / "overlay_quantile_sweep.csv", index=False, encoding="utf-8")

    # report
    report_lines = []
    for split in ["design", "eval"]:
        ev_sub = ev_decile[ev_decile["split"] == split]
        ov_sub = ov_decile[ov_decile["split"] == split]
        if not ev_sub.empty:
            worst_ev = ev_sub.loc[ev_sub["roi"].idxmin()]
            report_lines.append(f"{split} worst EV decile={int(worst_ev['decile'])} roi={worst_ev['roi']:.6f} stake={worst_ev['stake']:.0f}")
        if not ov_sub.empty:
            worst_ov = ov_sub.loc[ov_sub["roi"].idxmin()]
            report_lines.append(f"{split} worst |overlay| decile={int(worst_ov['decile'])} roi={worst_ov['roi']:.6f} stake={worst_ov['stake']:.0f}")
    (out_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"Wrote: {out_dir}")


if __name__ == "__main__":
    main()
