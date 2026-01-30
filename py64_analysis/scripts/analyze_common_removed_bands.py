from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


REQUIRED_KEYS = ["race_id", "horse_no", "asof_time"]
COL_CANDIDATES = {
    "race_id": ["race_id", "raceid", "race"],
    "horse_no": ["horse_no", "horse", "horse_num", "umaban"],
    "asof_time": ["asof_time", "asof_ts", "t_snap", "asof"],
    "stake": ["stake", "stake_yen", "stake_amount", "total_stake", "stake_sum"],
    "profit": ["profit", "net_profit", "pnl"],
    "odds": ["odds_at_buy", "odds", "buy_odds", "odds_buy"],
    "p_mkt": ["p_mkt", "market_prob"],
    "ev": ["ev", "ev_margin", "expected_value"],
    "overlay": ["overlay", "logit_resid", "market_offset", "delta_logit"],
}


def _warn(msg: str) -> None:
    print(f"[warn] {msg}", file=sys.stderr)


def _resolve_column(columns: Iterable[str], candidates: list[str]) -> Optional[str]:
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        col = lower_map.get(cand.lower())
        if col is not None:
            return col
    return None


def _parse_w_idx(folder_name: str) -> Optional[int]:
    if not folder_name.startswith("w"):
        return None
    try:
        return int(folder_name[1:4])
    except Exception:
        return None


def _infer_w_idx_offset(run_dir_name: str) -> int:
    name = run_dir_name.lower()
    if "w013_022" in name or "w013-022" in name:
        return 12
    return 0


def _find_bets_files(base_dir: Path) -> list[Path]:
    out: list[Path] = []
    for path in base_dir.rglob("bets.csv"):
        parent = path.parent.name
        if _parse_w_idx(parent) is None:
            continue
        out.append(path)
    return out


def _split_from_w_idx(w_idx: int) -> str:
    return "design" if w_idx <= 12 else "eval"


def _load_bets_from_dir(run_dir: Path, side: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    offset = _infer_w_idx_offset(run_dir.name)
    for bets_path in _find_bets_files(run_dir):
        w_idx_raw = _parse_w_idx(bets_path.parent.name)
        if w_idx_raw is None:
            continue
        w_idx = int(w_idx_raw) + int(offset)
        split = _split_from_w_idx(w_idx)

        df = pd.read_csv(bets_path)
        cols = list(df.columns)

        mapping: dict[str, str] = {}
        for key, candidates in COL_CANDIDATES.items():
            col = _resolve_column(cols, candidates)
            if col is not None:
                mapping[key] = col

        missing_keys = [k for k in REQUIRED_KEYS if k not in mapping]
        missing_metrics = [k for k in ["stake", "profit"] if k not in mapping]
        if missing_keys or missing_metrics:
            _warn(
                f"skip {bets_path} missing columns: "
                f"keys={missing_keys} metrics={missing_metrics}"
            )
            continue

        df = df.rename(columns={v: k for k, v in mapping.items()})
        for col in ["stake", "profit", "odds", "p_mkt", "ev", "overlay"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["race_id", "horse_no", "asof_time", "stake", "profit"])
        df["race_id"] = df["race_id"].astype(str)
        df["horse_no"] = df["horse_no"].astype(str)
        df["asof_time"] = df["asof_time"].astype(str)

        df["key"] = df["race_id"] + "|" + df["horse_no"] + "|" + df["asof_time"]
        df["w_idx"] = int(w_idx)
        df["split"] = split
        df["side"] = side
        frames.append(df)

    if not frames:
        return pd.DataFrame(
            columns=[
                "race_id",
                "horse_no",
                "asof_time",
                "stake",
                "profit",
                "odds",
                "p_mkt",
                "ev",
                "overlay",
                "key",
                "w_idx",
                "split",
                "side",
            ]
        )
    return pd.concat(frames, ignore_index=True)


def _summarize(df: pd.DataFrame) -> dict[str, float]:
    n_bets = int(len(df))
    stake = float(df["stake"].sum()) if n_bets > 0 else 0.0
    profit = float(df["profit"].sum()) if n_bets > 0 else 0.0
    roi = profit / stake if stake > 0 else float("nan")
    hit_rate = float((df["profit"] > 0).mean()) if n_bets > 0 else float("nan")
    avg_odds = float(df["odds"].mean()) if "odds" in df else float("nan")
    avg_ev = float(df["ev"].mean()) if "ev" in df else float("nan")
    avg_overlay = float(df["overlay"].mean()) if "overlay" in df else float("nan")
    return {
        "n_bets": n_bets,
        "stake": stake,
        "profit": profit,
        "roi": roi,
        "hit_rate": hit_rate,
        "avg_odds": avg_odds,
        "avg_ev": avg_ev,
        "avg_overlay": avg_overlay,
    }


def _metrics_by_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=group_cols + list(_summarize(df).keys()))
    out = df.groupby(group_cols, dropna=False).apply(lambda g: pd.Series(_summarize(g)))
    return out.reset_index()


def _assign_odds_band(series: pd.Series, max_odds: float) -> pd.Series:
    odds = pd.to_numeric(series, errors="coerce")
    bands = pd.Series(pd.NA, index=odds.index, dtype="object")
    bands[(odds >= 1.0) & (odds <= 5.0)] = "1-5"
    bands[(odds > 5.0) & (odds <= 10.0)] = "5-10"
    bands[(odds > 10.0) & (odds <= 20.0)] = "10-20"
    bands[(odds > 20.0) & (odds <= max_odds)] = f"20-{int(max_odds)}"
    return bands


def _decile_edges(values: pd.Series) -> Optional[np.ndarray]:
    v = pd.to_numeric(values, errors="coerce").dropna()
    if v.empty:
        return None
    qs = np.linspace(0, 1, 11)
    return np.quantile(v.to_numpy(), qs)


def _assign_decile(series: pd.Series, edges: Optional[np.ndarray]) -> pd.Series:
    if edges is None:
        return pd.Series(pd.NA, index=series.index)
    vals = pd.to_numeric(series, errors="coerce")
    idx = np.searchsorted(edges, vals, side="right") - 1
    idx = np.clip(idx, 0, 9)
    out = pd.Series(idx + 1, index=series.index, dtype="Int64")
    out[vals.isna()] = pd.NA
    return out


def _worst_band(df: pd.DataFrame, split: str, side: str) -> tuple[str, float]:
    subset = df[(df["split"] == split) & (df["side"] == side)].copy()
    subset = subset.dropna(subset=["roi"])
    if subset.empty:
        return "NA", float("nan")
    row = subset.loc[subset["roi"].idxmin()]
    return str(row.get("band", "NA")), float(row.get("roi", float("nan")))


def _worst_decile(df: pd.DataFrame, split: str, side: str) -> tuple[str, float]:
    subset = df[(df["split"] == split) & (df["side"] == side)].copy()
    subset = subset.dropna(subset=["roi"])
    if subset.empty:
        return "NA", float("nan")
    row = subset.loc[subset["roi"].idxmin()]
    return str(row.get("decile", "NA")), float(row.get("roi", float("nan")))


def _ensure_out_dir(out_dir: Path) -> Path:
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir
    if any(out_dir.iterdir()):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        sub = out_dir / ts
        sub.mkdir(parents=True, exist_ok=True)
        return sub
    return out_dir


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base-dirs", nargs="+", required=True)
    p.add_argument("--var-dirs", nargs="+", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--max-odds", type=float, default=30.0)
    args = p.parse_args()

    base_dirs = [Path(d) for d in args.base_dirs]
    var_dirs = [Path(d) for d in args.var_dirs]
    out_dir = _ensure_out_dir(Path(args.out_dir))

    base_df = pd.concat([_load_bets_from_dir(d, "base") for d in base_dirs], ignore_index=True)
    var_df = pd.concat([_load_bets_from_dir(d, "var") for d in var_dirs], ignore_index=True)

    rows_summary: list[dict[str, object]] = []
    bands_common_odds: list[pd.DataFrame] = []
    bands_removed_odds: list[pd.DataFrame] = []
    bands_common_pmkt: list[pd.DataFrame] = []
    bands_removed_pmkt: list[pd.DataFrame] = []
    bands_common_ev: list[pd.DataFrame] = []
    bands_removed_ev: list[pd.DataFrame] = []
    contrib_common_race: list[pd.DataFrame] = []
    contrib_removed_race: list[pd.DataFrame] = []
    contrib_common_bet: list[pd.DataFrame] = []
    contrib_removed_bet: list[pd.DataFrame] = []
    report_lines: list[str] = []

    for split in ["design", "eval"]:
        base_split = base_df[base_df["split"] == split].copy()
        var_split = var_df[var_df["split"] == split].copy()

        base_keys = set(base_split["key"].unique())
        var_keys = set(var_split["key"].unique())
        common_keys = base_keys & var_keys
        removed_keys = base_keys - var_keys
        added_keys = var_keys - base_keys

        base_common = base_split[base_split["key"].isin(common_keys)]
        var_common = var_split[var_split["key"].isin(common_keys)]
        base_removed = base_split[base_split["key"].isin(removed_keys)]
        var_added = var_split[var_split["key"].isin(added_keys)]

        for set_name, side, df in [
            ("common", "base", base_common),
            ("common", "var", var_common),
            ("removed", "base", base_removed),
            ("added", "var", var_added),
        ]:
            if df.empty:
                continue
            metrics = _summarize(df)
            rows_summary.append(
                {
                    "split": split,
                    "set": set_name,
                    "side": side,
                    **metrics,
                }
            )

        # Odds band tables
        for side_label, df in [("base", base_common), ("var", var_common)]:
            if df.empty or "odds" not in df:
                continue
            df = df.copy()
            df["band"] = _assign_odds_band(df["odds"], args.max_odds)
            df = df.dropna(subset=["band"])
            if df.empty:
                continue
            table = _metrics_by_group(df, ["split", "side", "band"])
            bands_common_odds.append(table)

        if not base_removed.empty and "odds" in base_removed:
            df = base_removed.copy()
            df["band"] = _assign_odds_band(df["odds"], args.max_odds)
            df = df.dropna(subset=["band"])
            if not df.empty:
                table = _metrics_by_group(df, ["split", "side", "band"])
                bands_removed_odds.append(table)

        # Decile bins based on base common
        pmkt_edges = _decile_edges(base_common["p_mkt"]) if "p_mkt" in base_common else None
        ev_edges = _decile_edges(base_common["ev"]) if "ev" in base_common else None

        for side_label, df in [("base", base_common), ("var", var_common)]:
            if df.empty:
                continue
            if "p_mkt" in df and pmkt_edges is not None:
                df_pmkt = df.copy()
                df_pmkt["decile"] = _assign_decile(df_pmkt["p_mkt"], pmkt_edges)
                df_pmkt = df_pmkt.dropna(subset=["decile"])
                if not df_pmkt.empty:
                    table = _metrics_by_group(df_pmkt, ["split", "side", "decile"])
                    bands_common_pmkt.append(table)
            if "ev" in df and ev_edges is not None:
                df_ev = df.copy()
                df_ev["decile"] = _assign_decile(df_ev["ev"], ev_edges)
                df_ev = df_ev.dropna(subset=["decile"])
                if not df_ev.empty:
                    table = _metrics_by_group(df_ev, ["split", "side", "decile"])
                    bands_common_ev.append(table)

        if not base_removed.empty:
            if "p_mkt" in base_removed and pmkt_edges is not None:
                df_pmkt = base_removed.copy()
                df_pmkt["decile"] = _assign_decile(df_pmkt["p_mkt"], pmkt_edges)
                df_pmkt = df_pmkt.dropna(subset=["decile"])
                if not df_pmkt.empty:
                    table = _metrics_by_group(df_pmkt, ["split", "side", "decile"])
                    bands_removed_pmkt.append(table)
            if "ev" in base_removed and ev_edges is not None:
                df_ev = base_removed.copy()
                df_ev["decile"] = _assign_decile(df_ev["ev"], ev_edges)
                df_ev = df_ev.dropna(subset=["decile"])
                if not df_ev.empty:
                    table = _metrics_by_group(df_ev, ["split", "side", "decile"])
                    bands_removed_ev.append(table)

        # Contribution tables
        def _contrib(df: pd.DataFrame, group_cols: list[str], top_n: int) -> pd.DataFrame:
            if df.empty:
                return pd.DataFrame(columns=["split", "side"] + group_cols + list(_summarize(df).keys()))
            table = _metrics_by_group(df, ["split", "side"] + group_cols)
            table = table.sort_values("profit", ascending=True).head(top_n)
            return table

        contrib_common_race.append(_contrib(base_common, ["race_id"], 50))
        contrib_common_race.append(_contrib(var_common, ["race_id"], 50))
        contrib_removed_race.append(_contrib(base_removed, ["race_id"], 50))

        contrib_common_bet.append(_contrib(base_common, ["race_id", "horse_no"], 100))
        contrib_common_bet.append(_contrib(var_common, ["race_id", "horse_no"], 100))
        contrib_removed_bet.append(_contrib(base_removed, ["race_id", "horse_no"], 100))

        # Report lines per split
        report_lines.append(f"## {split}")
        if not base_common.empty and not var_common.empty:
            base_common_metrics = _summarize(base_common)
            var_common_metrics = _summarize(var_common)
            report_lines.append(
                f"- common ROI: base={base_common_metrics['roi']:.4f}, "
                f"var={var_common_metrics['roi']:.4f}"
            )
        if not base_removed.empty:
            removed_metrics = _summarize(base_removed)
            report_lines.append(
                f"- removed ROI (base only): {removed_metrics['roi']:.4f} "
                f"(profit={removed_metrics['profit']:.0f})"
            )

    summary_df = pd.DataFrame(rows_summary)
    _write_csv(summary_df, out_dir / "overall_sets_summary.csv")

    def _cat_or_empty(frames: list[pd.DataFrame], cols: list[str]) -> pd.DataFrame:
        if not frames:
            return pd.DataFrame(columns=cols)
        return pd.concat(frames, ignore_index=True)

    _write_csv(
        _cat_or_empty(bands_common_odds, ["split", "side", "band"]),
        out_dir / "bands_common_by_odds.csv",
    )
    _write_csv(
        _cat_or_empty(bands_removed_odds, ["split", "side", "band"]),
        out_dir / "bands_removed_by_odds.csv",
    )
    _write_csv(
        _cat_or_empty(bands_common_pmkt, ["split", "side", "decile"]),
        out_dir / "bands_common_by_p_mkt_decile.csv",
    )
    _write_csv(
        _cat_or_empty(bands_removed_pmkt, ["split", "side", "decile"]),
        out_dir / "bands_removed_by_p_mkt_decile.csv",
    )
    _write_csv(
        _cat_or_empty(bands_common_ev, ["split", "side", "decile"]),
        out_dir / "bands_common_by_ev_decile.csv",
    )
    _write_csv(
        _cat_or_empty(bands_removed_ev, ["split", "side", "decile"]),
        out_dir / "bands_removed_by_ev_decile.csv",
    )

    _write_csv(
        _cat_or_empty(contrib_common_race, ["split", "side", "race_id"]),
        out_dir / "contrib_common_by_race.csv",
    )
    _write_csv(
        _cat_or_empty(contrib_removed_race, ["split", "side", "race_id"]),
        out_dir / "contrib_removed_by_race.csv",
    )
    _write_csv(
        _cat_or_empty(contrib_common_bet, ["split", "side", "race_id", "horse_no"]),
        out_dir / "contrib_common_by_bet.csv",
    )
    _write_csv(
        _cat_or_empty(contrib_removed_bet, ["split", "side", "race_id", "horse_no"]),
        out_dir / "contrib_removed_by_bet.csv",
    )

    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    # Required stdout lines
    common_odds = pd.read_csv(out_dir / "bands_common_by_odds.csv")
    removed_odds = pd.read_csv(out_dir / "bands_removed_by_odds.csv")
    common_ev = pd.read_csv(out_dir / "bands_common_by_ev_decile.csv")
    removed_ev = pd.read_csv(out_dir / "bands_removed_by_ev_decile.csv")

    for split in ["design", "eval"]:
        c_band, c_roi = _worst_band(common_odds, split, "var")
        c_ev, c_ev_roi = _worst_decile(common_ev, split, "var")
        r_band, r_roi = _worst_band(removed_odds, split, "base")
        r_ev, r_ev_roi = _worst_decile(removed_ev, split, "base")
        print(
            f"[{split}] common(var) worst odds band={c_band} roi={c_roi:.4f} "
            f"| worst EV decile={c_ev} roi={c_ev_roi:.4f}"
        )
        print(
            f"[{split}] removed(base) worst odds band={r_band} roi={r_roi:.4f} "
            f"| worst EV decile={r_ev} roi={r_ev_roi:.4f}"
        )

    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
