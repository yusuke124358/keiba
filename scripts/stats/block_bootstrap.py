#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_src_on_path() -> None:
    root = _repo_root()
    src = root / "py64_analysis" / "src"
    sys.path.insert(0, str(src))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, help="Path to variant per_bet_pnl.csv")
    p.add_argument("--baseline", required=True, help="Path to baseline per_bet_pnl.csv")
    p.add_argument("--out", required=True, help="Output JSON path (summary_stats.json)")
    p.add_argument("--B", type=int, default=2000, help="Bootstrap replicates")
    p.add_argument("--seed", type=int, default=0, help="Bootstrap seed")
    args = p.parse_args()

    _ensure_src_on_path()
    from keiba.stats.block_bootstrap import (  # type: ignore[import-not-found]
        BootstrapSettings,
        compute_block_bootstrap_summary,
        write_summary_json,
    )

    settings = BootstrapSettings(B=args.B, seed=args.seed)
    summary = compute_block_bootstrap_summary(args.variant, args.baseline, settings)
    write_summary_json(args.out, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
