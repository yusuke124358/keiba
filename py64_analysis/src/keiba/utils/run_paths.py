from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from ..config import PROJECT_ROOT


def now_ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_existing_dir(path: Path, what: str) -> Path:
    if not path.exists():
        raise SystemExit(f"{what} not found: {path}")
    if not path.is_dir():
        raise SystemExit(f"{what} is not a directory: {path}")
    return path


def parse_run_dir_arg(value: str) -> Path:
    return Path(value)


def make_analysis_out_dir(run_dir: Path, script_slug: str, ts: Optional[str] = None) -> Path:
    ts_val = ts or now_ts()
    return ensure_dir(run_dir / "analysis" / script_slug / ts_val)


def normalize_path_for_json(path: Path) -> str:
    try:
        if PROJECT_ROOT:
            return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except Exception:
        pass
    return str(path)
