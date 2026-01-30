"""
Inspect fact_race surface-related columns and values.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


def _ensure_import_path() -> None:
    project_root = Path(__file__).resolve().parents[2]
    src = project_root / "py64_analysis" / "src"
    if src.exists():
        sys.path.insert(0, str(src))


def _safe_col(name: str) -> bool:
    return bool(re.match(r"^[A-Za-z0-9_]+$", name))


def _get_columns(session) -> list[str]:
    from sqlalchemy import text

    rows = session.execute(
        text("SELECT column_name FROM information_schema.columns WHERE table_name='fact_race'")
    ).fetchall()
    return [r[0] for r in rows]


def _pick_date_col(cols: list[str]) -> Optional[str]:
    if "date" in cols:
        return "date"
    if "race_date" in cols:
        return "race_date"
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect fact_race surface columns")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _ensure_import_path()
    from keiba.db.loader import get_session
    from sqlalchemy import text

    session = get_session()
    cols = _get_columns(session)
    (out_dir / "fact_race_columns.txt").write_text("\n".join(cols) + "\n", encoding="utf-8")

    keywords = [
        "surface",
        "track",
        "course",
        "syubetu",
        "shubetsu",
        "shubetu",
        "baba",
        "shiba",
        "dirt",
        "turf",
        "grass",
        "jyo",
        "jyou",
    ]
    pattern = re.compile("|".join(re.escape(k) for k in keywords))
    candidates = [c for c in cols if pattern.search(c.lower())]

    date_col = _pick_date_col(cols)
    start = "2024-01-01"
    end = "2025-12-20"

    rows = []
    for col in candidates:
        if not _safe_col(col):
            continue
        if date_col:
            sql = text(
                f"SELECT {col} AS val, COUNT(*) AS n "
                f"FROM fact_race WHERE {date_col} BETWEEN :s AND :e "
                f"GROUP BY {col} ORDER BY n DESC LIMIT 20"
            )
            vals = session.execute(sql, {"s": start, "e": end}).fetchall()
        else:
            sql = text(
                f"SELECT {col} AS val, COUNT(*) AS n "
                f"FROM fact_race GROUP BY {col} ORDER BY n DESC LIMIT 20"
            )
            vals = session.execute(sql).fetchall()
        for val, n in vals:
            rows.append({"column": col, "value": val, "count": int(n)})

    cand_df = pd.DataFrame(rows)
    cand_df.to_csv(out_dir / "fact_race_surface_candidates.csv", index=False, encoding="utf-8")

    sample_cols = ["race_id"]
    if date_col:
        sample_cols.append(date_col)
    for c in ["track_code", "distance"]:
        if c in cols and c not in sample_cols:
            sample_cols.append(c)
    for c in candidates[:8]:
        if c not in sample_cols:
            sample_cols.append(c)
    sample_cols = [c for c in sample_cols if _safe_col(c)]

    if sample_cols:
        sql = text(
            f"SELECT {', '.join(sample_cols)} FROM fact_race "
            + (f"WHERE {date_col} BETWEEN :s AND :e " if date_col else "")
            + "ORDER BY race_id LIMIT 10"
        )
        params = {"s": start, "e": end} if date_col else {}
        rows = session.execute(sql, params).fetchall()
        sample_df = pd.DataFrame(rows, columns=sample_cols)
    else:
        sample_df = pd.DataFrame()
    sample_df.to_csv(out_dir / "fact_race_sample.csv", index=False, encoding="utf-8")

    cand_line = ",".join(candidates) if candidates else "none"
    print(f"[fact_race] surface_candidates={cand_line}")
    print(f"Wrote: {out_dir}")


if __name__ == "__main__":
    main()
