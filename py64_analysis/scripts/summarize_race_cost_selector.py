"""
Summarize race_cost cap selector decisions per window.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd

WIN_RE = re.compile(r"^w(\d{3})_")


def _infer_w_idx_offset(run_dir_name: str) -> int:
    name = run_dir_name.lower()
    if "w013_022" in name or "w013-022" in name:
        return 12
    return 0


def _offset_window_name(val: str, offset: int) -> str:
    m = WIN_RE.match(str(val))
    if not m:
        return str(val)
    idx = int(m.group(1)) + int(offset)
    return f"w{idx:03d}_{str(val)[4:]}"


def _parse_window_idx(val) -> Optional[int]:
    if pd.isna(val):
        return None
    m = re.search(r"w(\d+)", str(val))
    if m:
        return int(m.group(1))
    if str(val).isdigit():
        return int(val)
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize race cost cap selector decisions")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--variant", type=str, default="selector")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--design-max-idx", type=int, default=12)
    args = ap.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    variants = manifest.get("variants", {})
    if args.variant not in variants:
        raise SystemExit(f"variant not in manifest: {args.variant}")

    rows: list[dict] = []
    project_root = Path(__file__).resolve().parents[2]

    for run_dir in variants[args.variant]:
        run_path = Path(run_dir)
        if not run_path.is_absolute():
            run_path = (project_root / run_path).resolve()
        if not run_path.exists():
            continue
        offset = _infer_w_idx_offset(run_path.name)
        for selector_path in sorted(run_path.glob("w*/artifacts/race_cost_cap_selector.json")):
            window_dir = selector_path.parent.parent
            window_name = window_dir.name
            w_idx = _parse_window_idx(window_name)
            if w_idx is None:
                continue
            w_idx = int(w_idx) + int(offset)
            split = "design" if w_idx <= int(args.design_max_idx) else "eval"

            try:
                data = json.loads(selector_path.read_text(encoding="utf-8"))
            except Exception:
                continue

            base = data.get("baseline", {}) if isinstance(data, dict) else {}
            candidates = data.get("candidates", []) if isinstance(data, dict) else []
            rows.append(
                {
                    "window_name": _offset_window_name(window_name, offset),
                    "window_idx": w_idx,
                    "split": split,
                    "metric": data.get("metric") if isinstance(data, dict) else None,
                    "selected_cap": data.get("selected_cap") if isinstance(data, dict) else None,
                    "selected_reason": data.get("selected_reason") if isinstance(data, dict) else None,
                    "selected_from": data.get("selected_from") if isinstance(data, dict) else None,
                    "baseline_valid_roi": base.get("valid_roi"),
                    "baseline_valid_bets": base.get("valid_bets"),
                    "baseline_valid_stake": base.get("valid_stake"),
                    "baseline_valid_profit": base.get("valid_profit"),
                    "candidate_caps": json.dumps(data.get("candidate_caps"), ensure_ascii=False),
                    "candidates_json": json.dumps(candidates, ensure_ascii=False),
                    "error": data.get("error") if isinstance(data, dict) else None,
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "selector_decisions.csv", index=False, encoding="utf-8")

    if df.empty:
        counts = pd.DataFrame(columns=["split", "selected_cap", "count", "rate"])
    else:
        df_counts = df.copy()
        df_counts["selected_cap"] = df_counts["selected_cap"].apply(lambda v: "none" if pd.isna(v) else str(v))
        counts = (
            df_counts.groupby(["split", "selected_cap"], dropna=False)
            .size()
            .reset_index(name="count")
        )
        totals = counts.groupby("split")["count"].transform("sum")
        counts["rate"] = counts["count"] / totals

    counts.to_csv(out_dir / "selector_counts.csv", index=False, encoding="utf-8")


if __name__ == "__main__":
    main()
