from datetime import datetime

import pandas as pd
import pytest

from keiba.features.history_features import compute_entity_track_dist_win_rate_map


def test_track_dist_win_rate_map_is_time_causal_and_smoothed():
    asof = datetime(2025, 1, 1, 12, 0, 0)
    df = pd.DataFrame(
        [
            dict(
                entity_id="A",
                race_dt=datetime(2024, 12, 31, 11),
                track_code="01",
                distance=1600,
                finish_pos=1,
            ),
            dict(
                entity_id="A",
                race_dt=datetime(2024, 12, 30, 11),
                track_code="02",
                distance=1600,
                finish_pos=2,
            ),
            dict(
                entity_id="A",
                race_dt=datetime(2025, 1, 1, 13),
                track_code="01",
                distance=1600,
                finish_pos=1,
            ),  # future
            dict(
                entity_id="B",
                race_dt=datetime(2024, 12, 31, 10),
                track_code="02",
                distance=1600,
                finish_pos=1,
            ),
            dict(
                entity_id="B",
                race_dt=datetime(2024, 12, 30, 10),
                track_code="02",
                distance=1600,
                finish_pos=9,
            ),
        ]
    )
    out = compute_entity_track_dist_win_rate_map(
        df,
        asof_time=asof,
        track_code="01",
        target_distance=1600,
        entity_col="entity_id",
        smoothing_k=2.0,
    )
    assert out["A"] == pytest.approx(2 / 3, rel=1e-12)
    assert out["B"] == pytest.approx(0.5, rel=1e-12)
