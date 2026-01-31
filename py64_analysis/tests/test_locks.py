import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

from keiba.utils.locks import FileLock


def _make_tmp_dir() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    base = project_root / ".tmp" / f"locks_{uuid.uuid4().hex}"
    base.mkdir(parents=True, exist_ok=True)
    return base


def test_file_lock_acquire_release() -> None:
    base = _make_tmp_dir()
    try:
        lock_path = base / "db_write.lock"
        lock = FileLock(lock_path, timeout_seconds=0)
        lock.acquire()
        lock.release()
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_file_lock_conflict() -> None:
    base = _make_tmp_dir()
    try:
        lock_path = base / "db_write.lock"
        ready_path = base / "ready.txt"
        project_root = Path(__file__).resolve().parents[2]
        src = project_root / "py64_analysis" / "src"

        code = f"""
import sys, time
from pathlib import Path
sys.path.insert(0, r"{src}")
from keiba.utils.locks import FileLock
lock = FileLock(Path(r"{lock_path}"), timeout_seconds=0)
lock.acquire()
Path(r"{ready_path}").write_text("ready", encoding="utf-8")
time.sleep(2)
lock.release()
"""

        env = os.environ.copy()
        env["PYTHONPATH"] = f"{src}{os.pathsep}" + env.get("PYTHONPATH", "")
        proc = subprocess.Popen([sys.executable, "-c", code], env=env)
        try:
            deadline = time.time() + 5
            while time.time() < deadline and not ready_path.exists():
                time.sleep(0.05)
            assert ready_path.exists(), "child did not acquire lock"
            with pytest.raises(TimeoutError):
                FileLock(lock_path, timeout_seconds=0.1, poll_seconds=0.05).acquire()
        finally:
            proc.wait(timeout=5)

        lock = FileLock(lock_path, timeout_seconds=0.5, poll_seconds=0.05)
        lock.acquire()
        lock.release()
    finally:
        shutil.rmtree(base, ignore_errors=True)
