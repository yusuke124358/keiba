from datetime import datetime

import pandas as pd

from keiba.features.history_features import compute_horse_history_features


def test_history_features_no_future_leak():
    asof = datetime(2025, 1, 1, 12, 0, 0)
    df = pd.DataFrame(
        [
            {"race_dt": datetime(2025, 1, 1, 11, 0, 0), "finish_pos": 1, "distance": 1600, "last_3f": 35.0},
            # 未来（asofより後）→必ず除外される
            {"race_dt": datetime(2025, 1, 1, 13, 0, 0), "finish_pos": 1, "distance": 1600, "last_3f": 35.0},
        ]
    )

    feats = compute_horse_history_features(df, asof_time=asof, target_distance=1600)
    assert feats["horse_starts_365"] == 1
    assert feats["horse_wins_365"] == 1
    assert feats["horse_last_finish"] == 1


def test_history_features_first_start_is_empty():
    asof = datetime(2025, 1, 1, 12, 0, 0)
    df = pd.DataFrame([])
    feats = compute_horse_history_features(df, asof_time=asof, target_distance=1600)
    assert feats["horse_starts_365"] == 0
    assert feats["horse_wins_365"] == 0
    assert feats["horse_last_finish"] is None


