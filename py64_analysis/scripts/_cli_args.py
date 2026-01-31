from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

from keiba.utils.run_paths import make_analysis_out_dir, parse_run_dir_arg, require_existing_dir


def add_run_dir_arg(parser: ArgumentParser, *, required: bool = True, name: str = "--run-dir") -> None:
    parser.add_argument(name, required=required, type=parse_run_dir_arg, help="run_dir path (required)")


def add_out_dir_arg(parser: ArgumentParser, *, required: bool = False, name: str = "--out-dir") -> None:
    parser.add_argument(name, required=required, type=Path, default=None, help="output directory (optional)")


def resolve_out_dir(run_dir: Path, out_dir: Optional[Path], script_slug: str) -> Path:
    if out_dir is not None:
        return out_dir
    return make_analysis_out_dir(require_existing_dir(run_dir, "run_dir"), script_slug)
