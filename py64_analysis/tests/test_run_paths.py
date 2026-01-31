import re
import shutil
from pathlib import Path

from keiba.utils.run_paths import make_analysis_out_dir, now_ts


def test_now_ts_format() -> None:
    ts = now_ts()
    assert re.match(r"^\d{8}_\d{6}$", ts)


def test_make_analysis_out_dir() -> None:
    base_dir = Path(__file__).resolve().parent / "_tmp_run_paths"
    if base_dir.exists():
        shutil.rmtree(base_dir, ignore_errors=True)
    base_dir.mkdir(parents=True, exist_ok=True)
    try:
        run_dir = base_dir / "run"
        run_dir.mkdir()
        out_dir = make_analysis_out_dir(run_dir, "script_slug", ts="20260101_000000")
        assert out_dir == run_dir / "analysis" / "script_slug" / "20260101_000000"
        assert out_dir.exists()
    finally:
        shutil.rmtree(base_dir, ignore_errors=True)
