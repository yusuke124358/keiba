from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import yaml

from .metrics_schema import MetricsV01, validate_metrics_v01
from ..config import PROJECT_ROOT, get_data_path


def _project_root() -> Path:
    if PROJECT_ROOT:
        return PROJECT_ROOT
    return Path(__file__).resolve().parents[4]


def _rel_path(path: Path) -> str:
    root = _project_root()
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _sha256_file(path: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _git_commit() -> Optional[str]:
    try:
        root = _project_root()
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root))
        return out.decode("utf-8").strip()
    except Exception:
        return None


def _get_db_max_race_date() -> Optional[str]:
    try:
        from sqlalchemy import text
        from ..db.loader import get_session

        sess = get_session()
        try:
            row = sess.execute(text("SELECT MAX(date) FROM fact_race")).fetchone()
        finally:
            sess.close()
        if row and row[0]:
            return row[0].isoformat()
    except Exception:
        return None
    return None


def _get_raw_max_mtime() -> Optional[str]:
    try:
        raw_dir = get_data_path("data/raw")
        if not raw_dir.exists():
            return None
        files = list(raw_dir.rglob("*.jsonl"))
        if not files:
            return None
        max_mtime = max(p.stat().st_mtime for p in files)
        return datetime.fromtimestamp(max_mtime, tz=timezone.utc).isoformat()
    except Exception:
        return None


def _get_data_cutoff() -> Dict[str, Optional[str]]:
    return {
        "db_max_race_date": _get_db_max_race_date(),
        "raw_max_mtime": _get_raw_max_mtime(),
    }


def _safe_get(d: dict, path: str, default=None):
    cur = d
    for key in path.split("."):
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur


def _infer_prob_variant(config_path: Optional[Path]) -> str:
    if config_path is None or not config_path.exists():
        return "unknown"
    cfg = _read_yaml(config_path)
    betting = cfg.get("betting") if isinstance(cfg, dict) else {}
    if isinstance(betting, dict):
        enabled = betting.get("enable_market_blend")
        if enabled is None and isinstance(betting.get("market_blend"), dict):
            enabled = betting["market_blend"].get("enabled")
        if bool(enabled):
            return "p_blend"
    return "p_hat"


def _infer_market_prob_method(config_path: Optional[Path]) -> Optional[str]:
    if config_path is None or not config_path.exists():
        return None
    cfg = _read_yaml(config_path)
    betting = cfg.get("betting") if isinstance(cfg, dict) else {}
    if isinstance(betting, dict):
        val = betting.get("market_prob_method")
        if val is not None:
            return str(val)
    return None


def _infer_market_prob_mode(config_path: Optional[Path]) -> Optional[str]:
    if config_path is None or not config_path.exists():
        return None
    cfg = _read_yaml(config_path)
    model = cfg.get("model") if isinstance(cfg, dict) else {}
    if isinstance(model, dict):
        val = model.get("market_prob_mode")
        if val is not None:
            return str(val)
    return None


def _universe_from_config(config_path: Optional[Path]) -> Dict[str, Any]:
    if config_path is None or not config_path.exists():
        return {}
    cfg = _read_yaml(config_path)
    uni = cfg.get("universe") if isinstance(cfg, dict) else {}
    if not isinstance(uni, dict):
        return {}
    track_codes = sorted(str(x) for x in (uni.get("track_codes") or []))
    exclude_ids = [str(x) for x in (uni.get("exclude_race_ids") or [])]
    h = hashlib.sha256("\n".join(sorted(exclude_ids)).encode("utf-8")).hexdigest() if exclude_ids else None
    return {
        "track_codes": track_codes,
        "require_results": bool(uni.get("require_results")) if "require_results" in uni else None,
        "require_ts_win": bool(uni.get("require_ts_win")) if "require_ts_win" in uni else None,
        "exclude_race_ids_count": len(exclude_ids),
        "exclude_race_ids_hash": h,
    }


def _coerce_float(val: Any) -> Optional[float]:
    try:
        if val is None:
            return None
        return float(val)
    except Exception:
        return None


def _coerce_int(val: Any) -> Optional[int]:
    try:
        if val is None:
            return None
        return int(val)
    except Exception:
        return None


def _split_from_summary(summary: dict) -> Dict[str, Dict[str, Optional[Any]]]:
    out: Dict[str, Dict[str, Optional[Any]]] = {}
    for key in ["train", "valid", "test"]:
        block = summary.get(key, {})
        out[key] = {
            "start": block.get("start"),
            "end": block.get("end"),
            "n_races": block.get("n_races"),
        }
    return out


def _pick_step14_from_csv(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        import pandas as pd
    except Exception:
        return None, "pandas_unavailable"
    try:
        df = pd.read_csv(path)
    except Exception:
        return None, "read_failed"
    if df.empty:
        return None, "empty"

    if "split" in df.columns:
        eval_df = df[df["split"] == "eval"]
        if not eval_df.empty:
            df = eval_df

    if "strategy" in df.columns:
        strategies = list(df["strategy"].dropna().unique())
        if len(strategies) != 1:
            return None, "multiple_strategies"

    if len(df) != 1:
        return None, "multiple_rows"

    row = df.iloc[0]
    return {
        "roi": _coerce_float(row.get("roi")),
        "n_bets": _coerce_int(row.get("n_bets")),
        "total_stake": _coerce_float(row.get("total_stake")),
        "total_profit": _coerce_float(row.get("total_profit")),
        "max_drawdown": _coerce_float(row.get("max_drawdown")),
    }, None


def extract_metrics_from_holdout_run(
    run_dir: Path,
    data_cutoff_override: Optional[Dict[str, Optional[str]]] = None,
) -> Dict[str, Any]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found: {summary_path}")
    summary = _read_json(summary_path)

    config_path = run_dir / "config_used.yaml"
    config_rel = _rel_path(config_path) if config_path.exists() else None
    config_hash = _sha256_file(config_path) if config_path.exists() else None

    metrics = MetricsV01(
        run_kind="holdout",
        run_dir=_rel_path(run_dir),
        name=summary.get("name"),
        generated_at=summary.get("generated_at"),
        git_commit=_git_commit(),
        config_used_path=config_rel,
        config_hash_sha256=config_hash,
        data_cutoff=data_cutoff_override or _get_data_cutoff(),
        split=_split_from_summary(summary),
        universe=_universe_from_config(config_path if config_path.exists() else None),
        betting={
            "buy_t_minus_minutes": summary.get("buy_t_minus_minutes"),
            "closing_odds_multiplier": summary.get("closing_odds_multiplier"),
            "closing_odds_multiplier_estimated": summary.get("closing_odds_multiplier_estimated"),
            "closing_odds_multiplier_quantile": summary.get("closing_odds_multiplier_quantile"),
            "slippage_summary": summary.get("slippage_summary"),
            "slippage_table": summary.get("slippage_table"),
            "market_prob_method": _infer_market_prob_method(config_path if config_path.exists() else None),
            "market_prob_mode": _infer_market_prob_mode(config_path if config_path.exists() else None),
        },
        prob_variant_used=_infer_prob_variant(config_path if config_path.exists() else None),
        backtest={
            "n_bets": _coerce_int(_safe_get(summary, "backtest.n_bets")),
            "roi": _coerce_float(_safe_get(summary, "backtest.roi")),
            "total_stake": _coerce_float(_safe_get(summary, "backtest.total_stake")),
            "total_profit": _coerce_float(_safe_get(summary, "backtest.total_profit")),
            "max_drawdown": _coerce_float(_safe_get(summary, "backtest.max_drawdown")),
        },
        pred_quality={
            "logloss_market": _coerce_float(_safe_get(summary, "pred_quality.logloss_market")),
            "logloss_blend": _coerce_float(_safe_get(summary, "pred_quality.logloss_blend")),
            "logloss_calibrated": _coerce_float(_safe_get(summary, "pred_quality.logloss_calibrated")),
            "brier_market": _coerce_float(_safe_get(summary, "pred_quality.brier_market")),
            "brier_blend": _coerce_float(_safe_get(summary, "pred_quality.brier_blend")),
            "brier_calibrated": _coerce_float(_safe_get(summary, "pred_quality.brier_calibrated")),
        },
        calibration={
            "ece": _coerce_float(_safe_get(summary, "pred_quality.ece_calibrated")),
        },
        step14=None,
        incomparable_reasons=[],
    )
    return metrics.to_dict()


def extract_metrics_from_rolling_run(
    group_dir: Path,
    data_cutoff_override: Optional[Dict[str, Optional[str]]] = None,
) -> Dict[str, Any]:
    summary_csv = group_dir / "summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"summary.csv not found: {summary_csv}")
    import pandas as pd

    df = pd.read_csv(summary_csv)
    if df.empty:
        raise ValueError(f"summary.csv is empty: {summary_csv}")

    name_val = None
    if "name" in df.columns:
        name_val = df["name"].dropna().iloc[0] if df["name"].dropna().size > 0 else None
    if name_val is None:
        name_val = group_dir.name

    gen_val = None
    if "generated_at" in df.columns and df["generated_at"].dropna().size > 0:
        gen_val = str(df["generated_at"].dropna().iloc[0])

    def _min_date(col: str) -> Optional[str]:
        if col not in df.columns:
            return None
        s = pd.to_datetime(df[col], errors="coerce")
        if s.notna().any():
            return s.min().date().isoformat()
        return None

    def _max_date(col: str) -> Optional[str]:
        if col not in df.columns:
            return None
        s = pd.to_datetime(df[col], errors="coerce")
        if s.notna().any():
            return s.max().date().isoformat()
        return None

    def _unique_or_none(col: str) -> Optional[Any]:
        if col not in df.columns:
            return None
        vals = df[col].dropna().unique()
        if len(vals) == 1:
            val = vals[0]
            if hasattr(val, "item"):
                try:
                    val = val.item()
                except Exception:
                    pass
            return val
        return None

    def _mean_or_none(col: str) -> Optional[float]:
        if col not in df.columns:
            return None
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            return None
        return float(s.mean())

    backtest = {
        "n_bets": _coerce_int(df["n_bets"].sum()) if "n_bets" in df.columns else None,
        "total_stake": _coerce_float(df["total_stake"].sum()) if "total_stake" in df.columns else None,
        "total_profit": _coerce_float(df["total_profit"].sum()) if "total_profit" in df.columns else None,
        "max_drawdown": _coerce_float(df["max_drawdown"].max()) if "max_drawdown" in df.columns else None,
    }
    if (
        backtest.get("total_stake") is not None
        and backtest.get("total_profit") is not None
        and backtest.get("total_stake", 0) > 0
    ):
        backtest["roi"] = backtest["total_profit"] / backtest["total_stake"]
    else:
        backtest["roi"] = None

    config_path = None
    direct_cfg = group_dir / "config_used.yaml"
    if direct_cfg.exists():
        config_path = direct_cfg
    else:
        candidates = sorted(group_dir.glob("w*/config_used.yaml"))
        if candidates:
            config_path = candidates[0]
    config_rel = _rel_path(config_path) if config_path is not None else None
    config_hash = _sha256_file(config_path) if config_path is not None else None

    step14 = None
    step14_reason = None
    step14_path = group_dir / "walkforward_step14_summary.csv"
    if step14_path.exists():
        step14, step14_reason = _pick_step14_from_csv(step14_path)

    metrics = MetricsV01(
        run_kind="rolling_holdout",
        run_dir=_rel_path(group_dir),
        name=name_val,
        generated_at=gen_val,
        git_commit=_git_commit(),
        config_used_path=config_rel,
        config_hash_sha256=config_hash,
        data_cutoff=data_cutoff_override or _get_data_cutoff(),
        split={
            "train": {"start": _min_date("train_start"), "end": _max_date("train_end"), "n_races": _unique_or_none("n_train_races")},
            "valid": {"start": _min_date("valid_start"), "end": _max_date("valid_end"), "n_races": None},
            "test": {"start": _min_date("test_start"), "end": _max_date("test_end"), "n_races": _unique_or_none("n_test_races")},
        },
        universe=_universe_from_config(config_path),
        betting={
            "buy_t_minus_minutes": _unique_or_none("buy_t_minus_minutes"),
            "closing_odds_multiplier": _unique_or_none("closing_odds_multiplier"),
            "closing_odds_multiplier_estimated": _unique_or_none("closing_odds_multiplier_estimated"),
            "closing_odds_multiplier_quantile": _unique_or_none("closing_odds_multiplier_quantile"),
            "slippage_summary": None,
            "slippage_table": None,
            "market_prob_method": _infer_market_prob_method(config_path),
            "market_prob_mode": _infer_market_prob_mode(config_path),
        },
        prob_variant_used=_infer_prob_variant(config_path),
        backtest=backtest,
        pred_quality={
            "logloss_market": _mean_or_none("pred_logloss_market"),
            "logloss_blend": _mean_or_none("pred_logloss_blend"),
            "logloss_calibrated": _mean_or_none("pred_logloss_calibrated"),
            "brier_market": _mean_or_none("pred_brier_market"),
            "brier_blend": _mean_or_none("pred_brier_blend"),
            "brier_calibrated": _mean_or_none("pred_brier_calibrated"),
        },
        step14=step14,
        incomparable_reasons=[f"step14:{step14_reason}"] if step14_reason else [],
    )
    return metrics.to_dict()


def write_metrics_json(run_dir: Path, run_kind: str) -> Path:
    if run_kind == "holdout":
        metrics = extract_metrics_from_holdout_run(run_dir)
    elif run_kind == "rolling_holdout":
        metrics = extract_metrics_from_rolling_run(run_dir)
    else:
        raise ValueError(f"Unknown run_kind: {run_kind}")
    validate_metrics_v01(metrics)
    out_path = run_dir / "metrics.json"
    out_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path
