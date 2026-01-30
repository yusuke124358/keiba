"""
JV-Link RA LapTime diagnostic.

Purpose: confirm LapTime fields exist in real JV-Data records.
"""
from __future__ import annotations

import argparse
import csv
import os
import json
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Optional

from py32_fetcher.jvlink_client import JVLinkClient
from py32_fetcher.parsers import parse_record


def _s(raw_bytes: bytes, pos_1based: int, size: int) -> str:
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


def _extract_laps(raw_bytes: bytes) -> tuple[list[str], list[Optional[float]]]:
    laps_raw: list[str] = []
    laps_sec: list[Optional[float]] = []
    base_pos = 891
    for i in range(25):
        raw = _s(raw_bytes, base_pos + i * 3, 3)
        laps_raw.append(raw)
        laps_sec.append(_parse_time_tenths(raw))
    return laps_raw, laps_sec


def _scan_jsonl_sources(
    paths: list[Path], max_records: int, min_race_date: Optional[str]
) -> list[dict]:
    rows: list[dict] = []
    for root in paths:
        if not root.exists():
            continue
        for fp in root.rglob("RACE_*.jsonl"):
            try:
                with fp.open("r", encoding="utf-8") as f:
                    for line in f:
                        if "\"record_type\": \"RA\"" not in line:
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        race_id = obj.get("race_id") or ""
                        if min_race_date and len(race_id) >= 8:
                            if race_id[:8] < min_race_date:
                                continue
                        laps = obj.get("lap_times_200m")
                        if not isinstance(laps, list):
                            laps = []
                        nonnull_count = sum(1 for v in laps if v is not None)
                        row = {
                            "race_id": race_id,
                            "data_kubun": obj.get("data_kubun"),
                            "distance": obj.get("distance"),
                            "track_code": obj.get("track_code"),
                            "track_cd": obj.get("track_cd"),
                            "going_turf": obj.get("going_turf"),
                            "going_dirt": obj.get("going_dirt"),
                            "lap_nonnull_count": nonnull_count,
                        }
                        for i in range(5):
                            row[f"lap_raw_{i+1}"] = ""
                            row[f"lap_sec_{i+1}"] = laps[i] if i < len(laps) else None
                        rows.append(row)
                        if len(rows) >= max_records:
                            return rows
            except Exception:
                continue
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-records", type=int, default=200)
    parser.add_argument("--data-spec", default="RACE")
    parser.add_argument("--option", type=int, default=4)
    parser.add_argument("--from-time", default="20240101000000")
    parser.add_argument(
        "--software-id",
        default=os.environ.get("KEIBA_JV_SID")
        or os.environ.get("KEIBA_JV_SOFTWARE_ID")
        or "UNKNOWN",
    )
    parser.add_argument(
        "--service-key",
        default=os.environ.get("KEIBA_JV_USE_KEY") or None,
    )
    parser.add_argument(
        "--fallback-jsonl",
        action="store_true",
        help="When JVOpen fails, scan local JSONL files under data/raw and data/raw_backfill_*.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "ra_laptime_samples.csv"
    report_path = out_dir / "report.md"

    source = "jvlink"
    rows: list[dict] = []
    nonnull_counts: list[int] = []
    any_lap_nonnull = 0

    client = JVLinkClient(software_id=args.software_id, service_key=args.service_key)
    ret_code, read_count, dl_count, last_file_ts = client.open_stored(
        args.data_spec, args.from_time, option=args.option
    )
    if ret_code != 0:
        client.close()
        if not args.fallback_jsonl:
            raise RuntimeError(f"JVOpen failed: {ret_code}")
        source = f"jsonl_fallback (JVOpen={ret_code})"
        roots = [
            Path("data/raw_backfill_c4_2024"),
            Path("data/raw_backfill_c4_2025"),
            Path("data/raw"),
            Path("data/raw_backfill_c4_2023"),
            Path("data/raw_backfill_c4_2022"),
            Path("data/raw/c2_2020_2023"),
        ]
        min_race_date = args.from_time[:8] if args.from_time else None
        rows = _scan_jsonl_sources(roots, args.max_records, min_race_date)
        nonnull_counts = [r.get("lap_nonnull_count", 0) or 0 for r in rows]
        any_lap_nonnull = sum(1 for v in nonnull_counts if v > 0)
    else:
        while True:
            ret, data, filename = client.read()
            if ret == 0:
                break
            if ret == -1:
                continue
            if ret == -3:
                continue
            if ret < -1:
                client.close()
                raise RuntimeError(f"JVRead error: {ret}")

            record = parse_record(data, args.data_spec, filename)
            if not record or record.get("record_type") != "RA":
                continue

            laps_raw, laps_sec = _extract_laps(data)
            nonnull_count = sum(1 for v in laps_sec if v is not None)
            if nonnull_count > 0:
                any_lap_nonnull += 1
            nonnull_counts.append(nonnull_count)

            row = {
                "race_id": record.get("race_id"),
                "data_kubun": record.get("data_kubun"),
                "distance": record.get("distance"),
                "track_code": record.get("track_code"),
                "track_cd": record.get("track_cd"),
                "going_turf": record.get("going_turf"),
                "going_dirt": record.get("going_dirt"),
                "lap_nonnull_count": nonnull_count,
            }
            for i in range(5):
                row[f"lap_raw_{i+1}"] = laps_raw[i] if i < len(laps_raw) else ""
                row[f"lap_sec_{i+1}"] = laps_sec[i] if i < len(laps_sec) else None
            rows.append(row)

            if len(rows) >= args.max_records:
                break

        client.close()

    any_laptime_nonnull_ratio = 0.0
    median_nonnull_count = 0
    if rows:
        any_laptime_nonnull_ratio = any_lap_nonnull / len(rows)
        median_nonnull_count = int(median(nonnull_counts)) if nonnull_counts else 0

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        else:
            writer = csv.writer(f)
            writer.writerow(["note"])
            writer.writerow(["no RA records collected"])

    example = rows[0] if rows else {}
    for r in rows:
        if r.get("lap_nonnull_count", 0) > 0:
            example = r
            break
    example_laps_raw = [
        example.get("lap_raw_1"),
        example.get("lap_raw_2"),
        example.get("lap_raw_3"),
        example.get("lap_raw_4"),
        example.get("lap_raw_5"),
    ]
    if all(v in ("", None) for v in example_laps_raw):
        example_laps = [
            example.get("lap_sec_1"),
            example.get("lap_sec_2"),
            example.get("lap_sec_3"),
            example.get("lap_sec_4"),
            example.get("lap_sec_5"),
        ]
    else:
        example_laps = example_laps_raw

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# RA LapTime Diagnostic\n\n")
        f.write(f"- data_spec: {args.data_spec}\n")
        f.write(f"- option: {args.option}\n")
        f.write(f"- from_time: {args.from_time}\n")
        f.write(f"- source: {source}\n")
        f.write(f"- ra_samples: {len(rows)}\n")
        f.write(f"- any_laptime_nonnull_ratio: {any_laptime_nonnull_ratio:.4f}\n")
        f.write(f"- median_nonnull_count: {median_nonnull_count}\n")
        if example:
            f.write("\nExample (first5 raw laps):\n\n")
            f.write(f"- race_id: {example.get('race_id')}\n")
            f.write(f"- first5_laps: {example_laps}\n")

    print(
        f"[lapdiag] ra_samples={len(rows)} | any_laptime_nonnull_ratio={any_laptime_nonnull_ratio:.4f} | median_nonnull_count={median_nonnull_count}"
    )
    if example:
        print(
            f"[lapdiag] example race_id={example.get('race_id')} | first5_laps={example_laps}"
        )
    else:
        print("[lapdiag] example race_id=none | first5_laps=[]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
