from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import pandas as pd

from keiba.eval.extract_metrics import (
    extract_metrics_from_holdout_run,
    extract_metrics_from_rolling_run,
)


def _write_yaml(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _make_tmp_dir(name: str) -> Path:
    base = Path(__file__).resolve().parent / "_tmp"
    base.mkdir(parents=True, exist_ok=True)
    d = base / f"{name}_{uuid.uuid4().hex}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_extract_metrics_holdout() -> None:
    tmp_dir = _make_tmp_dir("holdout")
    run_dir = tmp_dir / "holdout_run"
    run_dir.mkdir(parents=True)

    summary = {
        "name": "sample_holdout",
        "generated_at": "20260101_000000",
        "train": {"start": "2020-01-01", "end": "2020-12-31", "n_races": 10, "features_saved": 0},
        "valid": {"start": "2021-01-01", "end": "2021-01-31", "n_races": 2, "features_saved": 0},
        "test": {"start": "2021-02-01", "end": "2021-02-28", "n_races": 3, "features_saved": 0},
        "buy_t_minus_minutes": 1,
        "backtest": {
            "n_bets": 10,
            "total_stake": 1000.0,
            "total_profit": 120.0,
            "roi": 0.12,
            "max_drawdown": 0.05,
        },
        "pred_quality": {
            "logloss_market": 0.2,
            "logloss_blend": 0.19,
            "logloss_calibrated": 0.21,
            "brier_market": 0.05,
            "brier_blend": 0.049,
            "brier_calibrated": 0.051,
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    _write_yaml(
        run_dir / "config_used.yaml",
        "universe:\n"
        "  track_codes: ['01','02']\n"
        "  require_results: true\n"
        "  require_ts_win: true\n"
        "  exclude_race_ids: ['2020010101010101']\n"
        "betting:\n"
        "  enable_market_blend: true\n"
        "  market_prob_method: p_mkt_col\n",
    )

    metrics = extract_metrics_from_holdout_run(
        run_dir,
        data_cutoff_override={"db_max_race_date": "2020-12-31", "raw_max_mtime": None},
    )

    assert metrics["run_kind"] == "holdout"
    assert metrics["backtest"]["roi"] == 0.12
    assert metrics["prob_variant_used"] == "p_blend"
    assert metrics["data_cutoff"]["db_max_race_date"] == "2020-12-31"
    assert metrics["universe"]["track_codes"] == ["01", "02"]
    assert metrics["betting"]["market_prob_method"] == "p_mkt_col"

    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_extract_metrics_rolling() -> None:
    tmp_dir = _make_tmp_dir("rolling")
    group_dir = tmp_dir / "rolling_group"
    group_dir.mkdir(parents=True)

    df = pd.DataFrame(
        [
            {
                "run_dir": "w001",
                "name": "rolling_sample",
                "generated_at": "20260101_000000",
                "train_start": "2020-01-01",
                "train_end": "2020-12-31",
                "valid_start": "2021-01-01",
                "valid_end": "2021-01-31",
                "test_start": "2021-02-01",
                "test_end": "2021-02-28",
                "buy_t_minus_minutes": 1,
                "closing_odds_multiplier": 1.0,
                "n_bets": 5,
                "roi": 0.1,
                "total_stake": 100.0,
                "total_profit": 10.0,
                "max_drawdown": 0.1,
                "pred_logloss_market": 0.2,
                "pred_logloss_blend": 0.19,
                "pred_logloss_calibrated": 0.21,
                "pred_brier_market": 0.05,
                "pred_brier_blend": 0.049,
                "pred_brier_calibrated": 0.051,
            },
            {
                "run_dir": "w002",
                "name": "rolling_sample",
                "generated_at": "20260101_000000",
                "train_start": "2020-02-01",
                "train_end": "2021-01-31",
                "valid_start": "2021-02-01",
                "valid_end": "2021-02-15",
                "test_start": "2021-03-01",
                "test_end": "2021-03-31",
                "buy_t_minus_minutes": 1,
                "closing_odds_multiplier": 1.0,
                "n_bets": 8,
                "roi": -0.1,
                "total_stake": 200.0,
                "total_profit": -20.0,
                "max_drawdown": 0.2,
                "pred_logloss_market": 0.21,
                "pred_logloss_blend": 0.2,
                "pred_logloss_calibrated": 0.22,
                "pred_brier_market": 0.051,
                "pred_brier_blend": 0.05,
                "pred_brier_calibrated": 0.052,
            },
        ]
    )
    df.to_csv(group_dir / "summary.csv", index=False)

    wdir = group_dir / "w001_20210101_20210130"
    wdir.mkdir(parents=True)
    _write_yaml(
        wdir / "config_used.yaml",
        "universe:\n"
        "  track_codes: ['01']\n"
        "  require_results: true\n"
        "  require_ts_win: true\n"
        "betting:\n"
        "  enable_market_blend: false\n",
    )

    metrics = extract_metrics_from_rolling_run(
        group_dir,
        data_cutoff_override={"db_max_race_date": "2021-03-31", "raw_max_mtime": None},
    )

    assert metrics["run_kind"] == "rolling_holdout"
    assert metrics["backtest"]["n_bets"] == 13
    assert metrics["backtest"]["total_stake"] == 300.0
    assert metrics["backtest"]["total_profit"] == -10.0
    assert metrics["backtest"]["max_drawdown"] == 0.2
    assert metrics["prob_variant_used"] == "p_hat"

    shutil.rmtree(tmp_dir, ignore_errors=True)
