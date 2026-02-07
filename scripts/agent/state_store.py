#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


@contextmanager
def file_lock(lock_path: Path, *, timeout_sec: float = 30.0) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    fd: int | None = None
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode("utf-8"))
            os.close(fd)
            fd = None
            break
        except FileExistsError:
            if time.time() - start > timeout_sec:
                raise RuntimeError(f"Timed out acquiring lock: {lock_path}")
            time.sleep(0.1)
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except Exception:
                    pass
                fd = None
    try:
        yield
    finally:
        try:
            lock_path.unlink()
        except Exception:
            pass


def default_state_paths() -> tuple[Path, Path]:
    base = (os.environ.get("KEIBA_STATE_DIR") or "").strip()
    if not base:
        raise RuntimeError("Set KEIBA_STATE_DIR for local scientist state.")
    base_dir = Path(base).expanduser()
    state_path = base_dir / "state.json"
    lock_path = base_dir / "lock"
    return state_path, lock_path


def load_state() -> dict[str, Any]:
    state_path, _ = default_state_paths()
    data = read_json(state_path)
    if "schema_version" not in data:
        data["schema_version"] = 1
    data.setdefault("items", {})
    if not isinstance(data.get("items"), dict):
        data["items"] = {}
    return data


def save_state(data: dict[str, Any]) -> None:
    state_path, _ = default_state_paths()
    data = dict(data)
    data["schema_version"] = 1
    data["updated_at"] = utc_now_iso()
    atomic_write_json(state_path, data)

