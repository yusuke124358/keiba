"""
Replay ROI candidate grid from existing rolling outputs (no retraining).
Reads bets.csv per window, applies rule grid, and summarizes step14 segments.
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WIN_RE = re.compile(r"^w\d{3}_(\d{8})_(\d{8})$")

DATE_COL_CANDIDATES = [
    ("race_date", "date"),
    ("event_date", "date"),
    ("date", "date"),
    ("settle_date", "date"),
    ("asof_date", "date"),
    ("asof_time", "datetime"),
    ("buy_time", "datetime"),
    ("bet_time", "datetime"),
]

STAKE_COL_CANDIDATES = [
    "stake",
    "stake_yen",
    "stake_amount",
    "bet_amount",
    "stake_raw",
]

RETURN_COL_CANDIDATES = [
    "return",
    "return_yen",
    "payout",
    "payout_yen",
]

PROFIT_COL_CANDIDATES = [
    "profit",
    "net_profit",
    "pnl",
]

ODDS_COL_CANDIDATES = [
    "odds_final",
    "odds_effective",
    "odds_at_buy",
    "odds",
]

EV_COL_CANDIDATES = [
    "ev_adj",
    "ev",
]


@dataclass(frozen=True)
class WindowSpec:
    name: str
    test_start: date
    test_end: date
    path: Path


def _parse_window_dates(name: str) -> Optional[tuple[date, date]]:
    m = WIN_RE.match(name)
    if not m:
        return None
    start = datetime.strptime(m.group(1), "%Y%m%d").date()
    end = datetime.strptime(m.group(2), "%Y%m%d").date()
    return start, end


def _detect_column(columns: list[str], candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in columns:
            return c
    return None


def _detect_date_column(columns: list[str]) -> tuple[Optional[str], Optional[str]]:
    for col, kind in DATE_COL_CANDIDATES:
        if col in columns:
            return col, kind
    return None, None


def _derive_date_from_race_id(df: pd.DataFrame) -> pd.Series:
    if "race_id" not in df.columns:
        return pd.Series([pd.NaT] * len(df))
    race_id = df["race_id"].astype(str).str.slice(0, 8)
    return pd.to_datetime(race_id, errors="coerce", format="%Y%m%d").dt.date


def _odds_band(val: float) -> str:
    if val < 3:
        return "<3"
    if val < 5:
        return "3-5"
    if val < 10:
        return "5-10"
    if val < 20:
        return "10-20"
    return "20+"


def _load_bets_for_window(window: WindowSpec, step_days: int) -> pd.DataFrame:
    bets_path = window.path / "bets.csv"
    if not bets_path.exists():
        raise SystemExit(f"bets.csv not found: {bets_path}")

    header = pd.read_csv(bets_path, nrows=0)
    cols = list(header.columns)

    date_col, date_kind = _detect_date_column(cols)
    stake_col = _detect_column(cols, STAKE_COL_CANDIDATES)
    profit_col = _detect_column(cols, PROFIT_COL_CANDIDATES)
    return_col = _detect_column(cols, RETURN_COL_CANDIDATES)
    odds_col = _detect_column(cols, ODDS_COL_CANDIDATES)
    ev_col = _detect_column(cols, EV_COL_CANDIDATES)

    if stake_col is None:
        raise SystemExit(f"Could not detect stake column in {bets_path}")
    if profit_col is None and return_col is None:
        raise SystemExit(f"Could not detect profit/return column in {bets_path}")

    usecols = [stake_col]
    if profit_col:
        usecols.append(profit_col)
    if return_col:
        usecols.append(return_col)
    if date_col:
        usecols.append(date_col)
    elif "race_id" in cols:
        usecols.append("race_id")
    if odds_col:
        usecols.append(odds_col)
    if ev_col:
        usecols.append(ev_col)
    if "p_hat" in cols:
        usecols.append("p_hat")

    df = pd.read_csv(bets_path, usecols=sorted(set(usecols)))

    if date_col is None:
        df["date"] = _derive_date_from_race_id(df)
    else:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.date

    segment_start = window.test_start
    segment_end = min(window.test_end, window.test_start + timedelta(days=step_days - 1))
    mask = (df["date"] >= segment_start) & (df["date"] <= segment_end)
    df = df.loc[mask].copy()

    df["stake"] = pd.to_numeric(df[stake_col], errors="coerce").fillna(0.0)
    if profit_col:
        df["profit"] = pd.to_numeric(df[profit_col], errors="coerce").fillna(0.0)
    else:
        df["profit"] = (
            pd.to_numeric(df[return_col], errors="coerce").fillna(0.0) - df["stake"]
        )

    if odds_col:
        df["odds_val"] = pd.to_numeric(df[odds_col], errors="coerce")
    else:
        df["odds_val"] = np.nan

    if ev_col:
        df["ev_val"] = pd.to_numeric(df[ev_col], errors="coerce")
    elif "p_hat" in df.columns and odds_col:
        df["ev_val"] = pd.to_numeric(df["p_hat"], errors="coerce") * df["odds_val"] - 1.0
    else:
        df["ev_val"] = np.nan

    df["odds_band"] = df["odds_val"].apply(lambda x: _odds_band(x) if pd.notna(x) else None)
    return df


def _apply_rule(df: pd.DataFrame, max_odds: Optional[float], exclude_bands: list[str], ev_threshold: float) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    if ev_threshold is not None:
        mask &= df["ev_val"].notna() & (df["ev_val"] >= ev_threshold)
    if max_odds is not None:
        mask &= df["odds_val"].notna() & (df["odds_val"] <= max_odds)
    if exclude_bands:
        mask &= df["odds_band"].notna() & (~df["odds_band"].isin(exclude_bands))
    return df.loc[mask].copy()


def _slug_exclude(bands: list[str]) -> str:
    if not bands:
        return "ex_none"
    mapping = {"<3": "lt3", "3-5": "3_5", "5-10": "5_10", "10-20": "10_20", "20+": "20p"}
    return "ex_" + "-".join(mapping.get(b, b) for b in bands)


def _resolve_run_dir(path: Path) -> Path:
    if path.exists():
        return path
    base = PROJECT_ROOT / "data" / "holdout_runs"
    candidate = base / path.name
    if candidate.exists():
        return candidate
    raise SystemExit(f"run_dir not found: {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay ROI candidate grid from rolling outputs (step14 segments)")
    ap.add_argument("--run_dir", type=Path, required=True)
    ap.add_argument("--out_dir_jp", type=Path, required=True)
    ap.add_argument("--out_dir_ascii", type=Path, required=True)
    ap.add_argument("--step_days", type=int, default=14)
    args = ap.parse_args()

    run_dir = _resolve_run_dir(args.run_dir)
    out_ascii = args.out_dir_ascii
    out_jp = args.out_dir_jp
    out_ascii.mkdir(parents=True, exist_ok=True)

    windows = []
    for p in run_dir.iterdir():
        if not p.is_dir():
            continue
        parsed = _parse_window_dates(p.name)
        if not parsed:
            continue
        windows.append(WindowSpec(name=p.name, test_start=parsed[0], test_end=parsed[1], path=p))
    windows.sort(key=lambda x: x.test_start)
    if not windows:
        raise SystemExit(f"No windows found under {run_dir}")

    bets_by_window = {}
    for w in windows:
        bets_by_window[w.name] = _load_bets_for_window(w, args.step_days)

    max_odds_list: list[Optional[float]] = [10, 15, 20, 30, None]
    exclude_bands_list = [
        [],
        ["20+"],
        ["5-10"],
        ["20+", "5-10"],
        ["20+", "5-10", "3-5"],
    ]
    ev_threshold_list = [0.0, 0.2]

    grid_rows = []
    by_window_rows = []
    rule_idx = 0

    for max_odds in max_odds_list:
        for exclude_bands in exclude_bands_list:
            for ev_thr in ev_threshold_list:
                rule_idx += 1
                max_part = f"max{int(max_odds)}" if max_odds is not None else "maxNone"
                ex_part = _slug_exclude(exclude_bands)
                ev_part = f"ev{str(ev_thr).replace('.', 'p')}"
                rule_id = f"r{rule_idx:03d}_{max_part}_{ex_part}_{ev_part}"

                total_stake = 0.0
                total_profit = 0.0
                total_bets = 0

                for w in windows:
                    base_df = bets_by_window[w.name]
                    filt = _apply_rule(base_df, max_odds=max_odds, exclude_bands=exclude_bands, ev_threshold=ev_thr)
                    stake = float(filt["stake"].sum())
                    profit = float(filt["profit"].sum())
                    n_bets = int(len(filt))
                    roi = (profit / stake) if stake > 0 else 0.0
                    by_window_rows.append(
                        {
                            "rule_id": rule_id,
                            "window_id": w.name,
                            "n_bets": n_bets,
                            "total_stake": stake,
                            "total_profit": profit,
                            "roi": roi,
                        }
                    )
                    total_stake += stake
                    total_profit += profit
                    total_bets += n_bets

                roi_total = (total_profit / total_stake) if total_stake > 0 else 0.0
                grid_rows.append(
                    {
                        "rule_id": rule_id,
                        "max_odds": max_odds if max_odds is not None else "",
                        "exclude_bands": "|".join(exclude_bands) if exclude_bands else "",
                        "ev_threshold": ev_thr,
                        "n_bets": total_bets,
                        "total_stake": total_stake,
                        "total_profit": total_profit,
                        "roi": roi_total,
                    }
                )

    grid_df = pd.DataFrame(grid_rows).sort_values("roi", ascending=False)
    grid_df.to_csv(out_ascii / "candidate_grid_overall.csv", index=False, encoding="utf-8")
    by_window_df = pd.DataFrame(by_window_rows)
    by_window_df.to_csv(out_ascii / "candidate_grid_by_window.csv", index=False, encoding="utf-8")

    top = grid_df.head(5)
    worst = grid_df.tail(3)
    report_lines = [
        "# replay ROI candidate grid (step14 segments)",
        "",
        f"- run_dir: {run_dir}",
        f"- windows: {len(windows)}",
        f"- step_days: {args.step_days}",
        "",
        "## Top candidates (by ROI)",
    ]
    if top.empty:
        report_lines.append("- no candidates")
    else:
        for _, row in top.iterrows():
            note = " (reference: n_bets<100)" if int(row["n_bets"]) < 100 else ""
            report_lines.append(
                f"- {row['rule_id']}: roi={row['roi']:.4f} "
                f"profit={row['total_profit']:.0f} stake={row['total_stake']:.0f} n_bets={int(row['n_bets'])}{note}"
            )

    report_lines.append("")
    report_lines.append("## Worst candidates (for sanity check)")
    for _, row in worst.iterrows():
        report_lines.append(
            f"- {row['rule_id']}: roi={row['roi']:.4f} "
            f"profit={row['total_profit']:.0f} stake={row['total_stake']:.0f} n_bets={int(row['n_bets'])}"
        )

    report_lines.append("")
    report_lines.append("ROI definition: ROI = profit / stake (profit = return - stake).")
    (out_ascii / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    stdout_lines = [
        "[audit] agents_md_updated=false | agents_md_zip_included=false",
        "[tests] pytest_passed=false",
        "[grid] replay_candidate_grid_done=true | best_rule_id=<TBD> | best_roi=<TBD>",
        f"[paths] out_dir={out_jp} | staging={out_ascii} | zip=<NOT_USED>",
    ]
    (out_ascii / "stdout_required.txt").write_text("\n".join(stdout_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
