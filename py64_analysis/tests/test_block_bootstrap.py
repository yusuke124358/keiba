from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from keiba.stats.block_bootstrap import BootstrapSettings, compute_block_bootstrap_summary


def _write_csv(path: Path, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def test_block_bootstrap_degenerate_ci(tmp_path: Path) -> None:
    variant = tmp_path / "variant.csv"
    baseline = tmp_path / "baseline.csv"

    # Two days with identical daily ROI for both arms -> bootstrap CI collapses.
    rows = [
        {
            "date": "2024-01-01",
            "race_id": "R1",
            "stake": 100,
            "return": 110,
            "profit": 10,
            "odds": 2.0,
        },
        {
            "date": "2024-01-02",
            "race_id": "R2",
            "stake": 100,
            "return": 110,
            "profit": 10,
            "odds": 2.0,
        },
    ]
    _write_csv(variant, rows)
    _write_csv(baseline, rows)

    settings = BootstrapSettings(B=500, seed=123)
    summary = compute_block_bootstrap_summary(variant, baseline, settings)
    stats = summary["stats"]

    assert stats["roi_ci95"] == pytest.approx([0.1, 0.1])
    assert stats["baseline_roi_ci95"] == pytest.approx([0.1, 0.1])
    assert stats["delta_roi_ci95"] == pytest.approx([0.0, 0.0])
    assert stats["p_one_sided_delta_le_0"] == pytest.approx(1.0)


def test_block_bootstrap_handles_missing_days(tmp_path: Path) -> None:
    variant = tmp_path / "variant.csv"
    baseline = tmp_path / "baseline.csv"

    # Variant has bets only on one of the days; baseline has both.
    _write_csv(
        variant,
        [
            {
                "date": "2024-01-02",
                "race_id": "R2",
                "stake": 100,
                "return": 105,
                "profit": 5,
                "odds": 10.0,
            },
        ],
    )
    _write_csv(
        baseline,
        [
            {
                "date": "2024-01-01",
                "race_id": "R1",
                "stake": 100,
                "return": 110,
                "profit": 10,
                "odds": 2.0,
            },
            {
                "date": "2024-01-02",
                "race_id": "R2",
                "stake": 100,
                "return": 100,
                "profit": 0,
                "odds": 10.0,
            },
        ],
    )

    settings = BootstrapSettings(B=300, seed=0)
    summary = compute_block_bootstrap_summary(variant, baseline, settings)

    assert "stats" in summary
    assert "breakdowns" in summary
    assert "odds_bucket" in summary["breakdowns"]
    assert "month" in summary["breakdowns"]
