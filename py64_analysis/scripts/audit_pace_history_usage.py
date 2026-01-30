"""
Audit whether pace-history features are used by the model and whether runs are identical.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Optional

import lightgbm as lgb  # noqa: F401  # required for unpickling Booster


PACE_FEATURES = [
    "horse_past_first3f_p20",
    "horse_past_last3f_p50",
    "horse_past_pace_diff_p50",
    "horse_past_lap_slope_p50",
    "horse_past_n_races_pace",
    "race_expected_first3f",
    "race_expected_last3f",
    "race_expected_pace_diff",
    "race_expected_pace_pressure",
]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _find_predictions_file(window_dir: Path) -> Optional[Path]:
    candidates = [
        window_dir / "predictions.csv",
        window_dir / "predictions.parquet",
        window_dir / "artifacts" / "predictions.csv",
        window_dir / "artifacts" / "predictions.parquet",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _load_model_features(model_path: Path) -> tuple[list[str], dict[str, float]]:
    if not model_path.exists():
        return [], {}
    with model_path.open("rb") as f:
        data = pickle.load(f)

    lgb_model = data.get("lgb_model")
    if lgb_model is None:
        names = data.get("feature_names") or []
        return list(names), {name: 0.0 for name in names}

    names = list(lgb_model.feature_name())
    gains = lgb_model.feature_importance(importance_type="gain")
    gains_map: dict[str, float] = {}
    for name, gain in zip(names, gains):
        gains_map[str(name)] = float(gain)
    return names, gains_map


def _safe_float(val: Any) -> Optional[float]:
    try:
        if val is None:
            return None
        return float(val)
    except Exception:
        return None


def _roi_triplet(summary: dict[str, Any]) -> tuple[Optional[float], Optional[float], Optional[float]]:
    bt = summary.get("backtest", {}) if isinstance(summary, dict) else {}
    roi = _safe_float(bt.get("roi"))
    stake = _safe_float(bt.get("total_stake"))
    profit = _safe_float(bt.get("total_profit"))
    return roi, stake, profit


def _float_equal(a: Optional[float], b: Optional[float], tol: float = 1e-12) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= tol


def _pace_gain_stats(names: list[str], gains: dict[str, float]) -> dict[str, Any]:
    total_gain = sum(float(v) for v in gains.values()) if gains else 0.0
    pace_gains = {k: gains.get(k, 0.0) for k in PACE_FEATURES if k in names}
    pace_gain_sum = sum(float(v) for v in pace_gains.values()) if pace_gains else 0.0
    pace_gain_frac = (pace_gain_sum / total_gain) if total_gain > 0 else 0.0
    top_pace_feat = ""
    top_pace_gain = 0.0
    if pace_gains:
        top_pace_feat = max(pace_gains, key=lambda k: pace_gains[k])
        top_pace_gain = float(pace_gains[top_pace_feat])
    return {
        "pace_feat_count": len(pace_gains),
        "pace_gain_sum": float(pace_gain_sum),
        "pace_gain_frac": float(pace_gain_frac),
        "top_pace_feat": top_pace_feat,
        "top_pace_gain": float(top_pace_gain),
        "has_any_pace_feat": len(pace_gains) > 0,
    }


def _top_features(names: list[str], gains: dict[str, float], top_n: int = 10) -> list[tuple[str, float]]:
    items = [(name, float(gains.get(name, 0.0))) for name in names]
    items.sort(key=lambda x: x[1], reverse=True)
    return items[:top_n]


def audit_runs(baseline_dir: Path, variant_dir: Path, out_dir: Path, label: str) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)

    def _list_windows(run_dir: Path) -> dict[str, Path]:
        windows = {p.name: p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("w")}
        if windows:
            return windows
        if (run_dir / "summary.json").exists():
            return {"__single__": run_dir}
        return {}

    base_windows = _list_windows(baseline_dir)
    var_windows = _list_windows(variant_dir)
    window_names = sorted(set(base_windows.keys()) & set(var_windows.keys()))
    if not window_names:
        raise SystemExit("No matching windows found between baseline and variant.")

    rows: list[dict[str, Any]] = []
    pace_gain_rows: list[dict[str, Any]] = []
    top_feat_rows: list[dict[str, Any]] = []

    for wname in window_names:
        bdir = base_windows[wname]
        vdir = var_windows[wname]

        b_bets = bdir / "bets.csv"
        v_bets = vdir / "bets.csv"
        bets_hash_equal = False
        b_bets_hash = ""
        v_bets_hash = ""
        if b_bets.exists() and v_bets.exists():
            b_bets_hash = _sha256_file(b_bets)
            v_bets_hash = _sha256_file(v_bets)
            bets_hash_equal = b_bets_hash == v_bets_hash

        b_summary = _read_json(bdir / "summary.json")
        v_summary = _read_json(vdir / "summary.json")
        b_roi, b_stake, b_profit = _roi_triplet(b_summary)
        v_roi, v_stake, v_profit = _roi_triplet(v_summary)
        roi_equal = _float_equal(b_roi, v_roi) and _float_equal(b_stake, v_stake) and _float_equal(b_profit, v_profit)

        b_pred = _find_predictions_file(bdir)
        v_pred = _find_predictions_file(vdir)
        preds_hash_equal = False
        b_pred_hash = ""
        v_pred_hash = ""
        if b_pred is not None and v_pred is not None:
            b_pred_hash = _sha256_file(b_pred)
            v_pred_hash = _sha256_file(v_pred)
            preds_hash_equal = b_pred_hash == v_pred_hash

        b_model_path = bdir / "artifacts" / "model.pkl"
        v_model_path = vdir / "artifacts" / "model.pkl"
        b_features, b_gains = _load_model_features(b_model_path)
        v_features, v_gains = _load_model_features(v_model_path)

        b_stats = _pace_gain_stats(b_features, b_gains)
        v_stats = _pace_gain_stats(v_features, v_gains)

        featurelist_equal = b_features == v_features

        window_label = wname
        if wname == "__single__":
            b_name = _read_json(bdir / "summary.json").get("name")
            v_name = _read_json(vdir / "summary.json").get("name")
            if b_name and v_name and b_name != v_name:
                window_label = f"{b_name}|{v_name}"
            else:
                window_label = b_name or v_name or bdir.name

        rows.append(
            {
                "window_name": window_label,
                "bets_hash_equal": bets_hash_equal,
                "roi_equal": roi_equal,
                "preds_hash_equal": preds_hash_equal if b_pred or v_pred else "",
                "featurelist_equal": featurelist_equal,
                "base_n_features": len(b_features),
                "var_n_features": len(v_features),
                "base_has_any_pace_feat": b_stats["has_any_pace_feat"],
                "var_has_any_pace_feat": v_stats["has_any_pace_feat"],
                "var_pace_feat_count": v_stats["pace_feat_count"],
                "var_pace_gain_sum": v_stats["pace_gain_sum"],
                "var_pace_gain_frac": v_stats["pace_gain_frac"],
                "top_pace_feat": v_stats["top_pace_feat"],
                "top_pace_gain": v_stats["top_pace_gain"],
                "base_bets_hash": b_bets_hash,
                "var_bets_hash": v_bets_hash,
                "base_roi": b_roi,
                "var_roi": v_roi,
                "base_pred_hash": b_pred_hash,
                "var_pred_hash": v_pred_hash,
            }
        )

        for feat in PACE_FEATURES:
            pace_gain_rows.append(
                {
                    "window_name": wname,
                    "side": "base",
                    "feature": feat,
                    "present": feat in b_features,
                    "gain": float(b_gains.get(feat, 0.0)),
                }
            )
            pace_gain_rows.append(
                {
                    "window_name": wname,
                    "side": "var",
                    "feature": feat,
                    "present": feat in v_features,
                    "gain": float(v_gains.get(feat, 0.0)),
                }
            )

        for side, names, gains in [("base", b_features, b_gains), ("var", v_features, v_gains)]:
            for feat, gain in _top_features(names, gains, top_n=10):
                top_feat_rows.append(
                    {
                        "window_name": wname,
                        "side": side,
                        "feature": feat,
                        "gain": float(gain),
                    }
                )

    audit_path = out_dir / "window_audit.csv"
    with audit_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    pace_path = out_dir / "pace_feature_gains.csv"
    with pace_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["window_name", "side", "feature", "present", "gain"])
        writer.writeheader()
        writer.writerows(pace_gain_rows)

    top_path = out_dir / "top_features.csv"
    with top_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["window_name", "side", "feature", "gain"])
        writer.writeheader()
        writer.writerows(top_feat_rows)

    def _windows_where(pred):
        return [r["window_name"] for r in rows if pred(r)]

    bets_all_equal = all(bool(r["bets_hash_equal"]) for r in rows)
    featurelists_all_equal = all(bool(r["featurelist_equal"]) for r in rows)
    var_has_any_rate = sum(1 for r in rows if r["var_has_any_pace_feat"]) / len(rows)
    var_pace_gain_nonzero_rate = sum(1 for r in rows if float(r["var_pace_gain_sum"]) > 0) / len(rows)

    summary_lines = [
        f"# Pace History Audit ({label})",
        "",
        f"- baseline: {baseline_dir}",
        f"- variant: {variant_dir}",
        f"- windows: {len(rows)}",
        f"- bets_all_equal: {bets_all_equal}",
        f"- model_featurelists_all_equal: {featurelists_all_equal}",
        f"- var_has_any_pace_feat_rate: {var_has_any_rate:.3f}",
        f"- var_pace_gain_nonzero_rate: {var_pace_gain_nonzero_rate:.3f}",
        "",
        "## Match layers",
        f"1) bets identical: {len(_windows_where(lambda r: r['bets_hash_equal']))} / {len(rows)}",
        "   windows: " + ", ".join(_windows_where(lambda r: r["bets_hash_equal"])) if _windows_where(lambda r: r["bets_hash_equal"]) else "   windows: none",
        f"2) feature list identical: {len(_windows_where(lambda r: r['featurelist_equal']))} / {len(rows)}",
        "   windows: " + ", ".join(_windows_where(lambda r: r["featurelist_equal"])) if _windows_where(lambda r: r["featurelist_equal"]) else "   windows: none",
        f"3) pace features missing: {len(_windows_where(lambda r: not r['var_has_any_pace_feat']))} / {len(rows)}",
        "   windows: " + ", ".join(_windows_where(lambda r: not r["var_has_any_pace_feat"])) if _windows_where(lambda r: not r["var_has_any_pace_feat"]) else "   windows: none",
        f"4) pace features present but gain=0: {len(_windows_where(lambda r: r['var_has_any_pace_feat'] and float(r['var_pace_gain_sum']) <= 0))} / {len(rows)}",
        "   windows: " + ", ".join(_windows_where(lambda r: r["var_has_any_pace_feat"] and float(r["var_pace_gain_sum"]) <= 0)) if _windows_where(lambda r: r["var_has_any_pace_feat"] and float(r["var_pace_gain_sum"]) <= 0) else "   windows: none",
        "",
        "Artifacts:",
        f"- window_audit.csv: {audit_path}",
        f"- pace_feature_gains.csv: {pace_path}",
        f"- top_features.csv: {top_path}",
    ]
    summary_path = out_dir / "summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    return {
        "windows": len(rows),
        "bets_all_equal": bets_all_equal,
        "featurelists_all_equal": featurelists_all_equal,
        "var_has_any_rate": var_has_any_rate,
        "var_pace_gain_nonzero_rate": var_pace_gain_nonzero_rate,
        "out_dir": str(out_dir),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit pace-history usage across runs.")
    ap.add_argument("--baseline-run-dir", required=True)
    ap.add_argument("--variant-run-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    baseline_dir = Path(args.baseline_run_dir)
    variant_dir = Path(args.variant_run_dir)
    out_dir = Path(args.out_dir)
    label = args.label or "run"

    result = audit_runs(baseline_dir, variant_dir, out_dir, label)
    print(
        f"[audit {label}] windows={result['windows']} | bets_all_equal={result['bets_all_equal']} | "
        f"model_featurelists_all_equal={result['featurelists_all_equal']} | "
        f"var_has_any_pace_feat_rate={result['var_has_any_rate']:.3f} | "
        f"var_pace_gain_nonzero_rate={result['var_pace_gain_nonzero_rate']:.3f}"
    )
    print(f"audit_dir={result['out_dir']}")


if __name__ == "__main__":
    main()
