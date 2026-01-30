from __future__ import annotations

import json
import math
from typing import Iterable, Optional


def _to_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _s_bytes(raw_bytes: bytes, pos_1based: int, size: int) -> str:
    a = int(pos_1based) - 1
    b = a + int(size)
    if a < 0 or b <= 0 or b > len(raw_bytes):
        return ""
    return raw_bytes[a:b].decode("shift_jis", errors="ignore")


def _parse_time_tenths(raw: str) -> Optional[float]:
    t = (raw or "").strip()
    if not t or not t.isdigit():
        return None
    try:
        v = int(t)
    except ValueError:
        return None
    if v <= 0:
        return None
    if v >= 999:
        return None
    return float(v) / 10.0


def extract_lap_times_200m_from_ra_bytes(raw_bytes: bytes) -> Optional[list[Optional[float]]]:
    if len(raw_bytes) < 891 + (3 * 25):
        return None
    laps: list[Optional[float]] = []
    base_pos = 891
    for i in range(25):
        raw = _s_bytes(raw_bytes, base_pos + i * 3, 3)
        laps.append(_parse_time_tenths(raw))
    if not any(v is not None for v in laps):
        return None
    return laps


def normalize_lap_times(lap_times) -> list[float]:
    if lap_times is None:
        return []
    if isinstance(lap_times, str):
        try:
            lap_times = json.loads(lap_times)
        except json.JSONDecodeError:
            return []
    if not isinstance(lap_times, (list, tuple)):
        return []
    out: list[float] = []
    for v in lap_times:
        fv = _to_float(v)
        if fv is None:
            continue
        out.append(fv)
    return out


def sum_first_n(lap_times, n: int) -> Optional[float]:
    times = normalize_lap_times(lap_times)
    if len(times) < n:
        return None
    return float(sum(times[:n]))


def sum_last_n(lap_times, n: int) -> Optional[float]:
    times = normalize_lap_times(lap_times)
    if len(times) < n:
        return None
    return float(sum(times[-n:]))


def lap_stats(lap_times) -> tuple[Optional[float], Optional[float], Optional[float]]:
    times = normalize_lap_times(lap_times)
    if len(times) < 2:
        return None, None, None
    n = len(times)
    mean = float(sum(times) / n)
    var = sum((t - mean) ** 2 for t in times) / n
    std = math.sqrt(var) if var >= 0 else None
    x_mean = (n - 1) / 2.0
    var_x = sum((i - x_mean) ** 2 for i in range(n)) / n
    slope = None
    if var_x > 0:
        cov = sum((i - x_mean) * (t - mean) for i, t in enumerate(times)) / n
        slope = cov / var_x
    return mean, std, slope


def pace_diff(pace_first3f, pace_last3f) -> Optional[float]:
    p1 = _to_float(pace_first3f)
    p2 = _to_float(pace_last3f)
    if p1 is None or p2 is None:
        return None
    return p1 - p2


def derive_pace_stats_from_laps(lap_times) -> dict:
    pace_first3f = sum_first_n(lap_times, 3)
    pace_last3f = sum_last_n(lap_times, 3)
    pace_diff_sec = pace_diff(pace_first3f, pace_last3f)
    lap_mean_sec, lap_std_sec, lap_slope = lap_stats(lap_times)
    return {
        "pace_first3f": pace_first3f,
        "pace_last3f": pace_last3f,
        "pace_diff_sec": pace_diff_sec,
        "lap_mean_sec": lap_mean_sec,
        "lap_std_sec": lap_std_sec,
        "lap_slope": lap_slope,
    }
