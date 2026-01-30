"""Audit odds_dyn_ev_margin score coverage and binding on eval windows."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


WIN_RE = re.compile(r"^w(\d{3})_(\d{8})_(\d{8})$")


def _read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        import yaml
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}


def _col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _stats(series: pd.Series) -> dict[str, Any]:
    s = series.dropna().astype(float)
    if s.empty:
        return {
            "nonnull_rate": 0.0,
            "mean": None,
            "std": None,
            "p1": None,
            "p50": None,
            "p90": None,
            "p99": None,
            "unique_values": 0,
            "constant": True,
        }
    vals = s.to_numpy()
    uniq = len({f"{v:.6f}" for v in vals})
    std = float(np.std(vals)) if len(vals) else None
    return {
        "nonnull_rate": float(len(vals)) / float(len(series)),
        "mean": float(np.mean(vals)),
        "std": std,
        "p1": float(np.quantile(vals, 0.01)),
        "p50": float(np.quantile(vals, 0.50)),
        "p90": float(np.quantile(vals, 0.90)),
        "p99": float(np.quantile(vals, 0.99)),
        "unique_values": int(uniq),
        "constant": bool(uniq <= 1 or (std is not None and std == 0.0)),
    }


def _load_bets(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _window_stats(run_dir: Path, label: str) -> list[dict[str, Any]]:
    rows = []
    for w in sorted(run_dir.iterdir()):
        if not w.is_dir():
            continue
        if not WIN_RE.match(w.name):
            continue
        bets_path = w / "bets.csv"
        df = _load_bets(bets_path)
        if df.empty:
            continue
        score_col = _col(df, ["odds_dyn_ev_score"])
        min_ev_eff_col = _col(df, ["min_ev_eff"])
        min_ev_eff_dyn_col = _col(df, ["min_ev_eff_odds_dyn"])
        passed_col = _col(df, ["passed_odds_dyn_ev_margin"])
        stake_col = _col(df, ["stake", "stake_yen", "stake_amount"])

        cfg = _read_yaml(w / "config_used.yaml")
        odm = (cfg.get("betting") or {}).get("odds_dyn_ev_margin") or {}
        meta = {
            "window": w.name,
            "run": label,
            "n_bets": int(len(df)),
            "odm_enabled": bool(odm.get("enabled", False)),
            "odm_metric": odm.get("metric"),
            "odm_lookback": odm.get("lookback_minutes"),
            "odm_direction": odm.get("direction"),
            "odm_ref": odm.get("ref"),
            "odm_slope": odm.get("slope"),
        }

        if score_col:
            stats = _stats(df[score_col])
            for k, v in stats.items():
                meta[f"score_{k}"] = v
        else:
            meta["score_nonnull_rate"] = 0.0
            meta["score_constant"] = True

        if min_ev_eff_col:
            stats = _stats(df[min_ev_eff_col])
            for k, v in stats.items():
                meta[f"min_ev_eff_{k}"] = v

        if min_ev_eff_dyn_col:
            stats = _stats(df[min_ev_eff_dyn_col])
            for k, v in stats.items():
                meta[f"min_ev_eff_dyn_{k}"] = v

        if passed_col:
            passed = df[passed_col].fillna(True)
            passed = passed.astype(bool)
            filtered = (~passed).sum()
            meta["passed_true"] = int(passed.sum())
            meta["passed_false"] = int(filtered)
            if stake_col:
                meta["filtered_stake"] = float(df.loc[~passed, stake_col].sum())
            else:
                meta["filtered_stake"] = None
        else:
            meta["passed_true"] = None
            meta["passed_false"] = None
            meta["filtered_stake"] = None

        rows.append(meta)
    return rows


def _hash_rows(df: pd.DataFrame, key_cols: list[str]) -> tuple[int, str]:
    df = df.copy()
    for c in key_cols:
        if c not in df.columns:
            return 0, ""
    df["key"] = df[key_cols].astype(str).agg("|".join, axis=1)
    values = sorted(df["key"].tolist())
    if not values:
        return 0, ""
    return len(values), str(hash("".join(values)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True)
    ap.add_argument("--var-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--diff-window", default=None)
    args = ap.parse_args()

    base_dir = Path(args.base_dir)
    var_dir = Path(args.var_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_rows = _window_stats(base_dir, "base")
    var_rows = _window_stats(var_dir, "var")
    df = pd.DataFrame(base_rows + var_rows)
    if not df.empty:
        df.to_csv(out_dir / "coverage_by_window.csv", index=False, encoding="utf-8")

    windows = sorted({r["window"] for r in base_rows} & {r["window"] for r in var_rows})
    diff_rows = []
    for win in windows:
        bdf = _load_bets(base_dir / win / "bets.csv")
        vdf = _load_bets(var_dir / win / "bets.csv")
        if bdf.empty or vdf.empty:
            continue
        key_cols = ["race_id", "horse_no", "asof_time"]
        if any(c not in bdf.columns for c in key_cols) or any(c not in vdf.columns for c in key_cols):
            continue
        bdf["key"] = bdf[key_cols].astype(str).agg("|".join, axis=1)
        vdf["key"] = vdf[key_cols].astype(str).agg("|".join, axis=1)
        b_keys = set(bdf["key"].tolist())
        v_keys = set(vdf["key"].tolist())
        filtered = b_keys - v_keys
        added = v_keys - b_keys
        b_stake = bdf.set_index("key")["stake"].to_dict() if "stake" in bdf.columns else {}
        v_stake = vdf.set_index("key")["stake"].to_dict() if "stake" in vdf.columns else {}
        diff_rows.append(
            {
                "window": win,
                "base_n_bets": int(len(b_keys)),
                "var_n_bets": int(len(v_keys)),
                "filtered_bets": int(len(filtered)),
                "added_bets": int(len(added)),
                "filtered_stake": float(sum(b_stake.get(k, 0.0) for k in filtered)),
                "added_stake": float(sum(v_stake.get(k, 0.0) for k in added)),
                "identical_passset": bool(len(filtered) == 0 and len(added) == 0),
            }
        )
    if diff_rows:
        pd.DataFrame(diff_rows).to_csv(out_dir / "diff_by_window.csv", index=False, encoding="utf-8")
    diff_win = args.diff_window or (windows[0] if windows else None)
    diff_out = {"status": "missing", "window": diff_win}
    if diff_win:
        bdf = _load_bets(base_dir / diff_win / "bets.csv")
        vdf = _load_bets(var_dir / diff_win / "bets.csv")
        key_cols = ["race_id", "horse_no", "asof_time", "stake", "odds_at_buy"]
        b_n, b_hash = _hash_rows(bdf, key_cols)
        v_n, v_hash = _hash_rows(vdf, key_cols)
        diff_out = {
            "status": "ok",
            "window": diff_win,
            "base_rows": int(b_n),
            "var_rows": int(v_n),
            "hash_match": bool(b_hash == v_hash) if b_hash and v_hash else False,
        }
    (out_dir / "diff_window_audit.json").write_text(
        json.dumps(diff_out, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Simple report
    lines = ["# odds_dyn_ev_margin coverage audit", ""]
    if df.empty:
        lines.append("- No window data found.")
    else:
        for run in ("base", "var"):
            sub = df[df["run"] == run]
            lines.append(f"## {run}")
            lines.append(f"- windows={len(sub)}")
            nonnull = float(sub["score_nonnull_rate"].median()) if "score_nonnull_rate" in sub else 0.0
            const_rate = float((sub["score_constant"] == True).mean()) if "score_constant" in sub else 1.0
            lines.append(f"- score_nonnull_rate_median={nonnull:.4f}")
            lines.append(f"- score_constant_window_rate={const_rate:.4f}")
            passed_false = sub["passed_false"].fillna(0).sum() if "passed_false" in sub else 0
            lines.append(f"- filtered_bets_sum={int(passed_false)}")
            lines.append("")

    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
