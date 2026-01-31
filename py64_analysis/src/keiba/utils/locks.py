from __future__ import annotations

import os
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path


class FileLock:
    def __init__(self, path: Path, timeout_seconds: float = 0, poll_seconds: float = 0.25) -> None:
        self.path = Path(path)
        self.timeout_seconds = float(timeout_seconds)
        self.poll_seconds = float(poll_seconds)
        self._fh = None
        self._locked = False

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        while True:
            try:
                self._fh = open(self.path, "a+b")
                if self.path.stat().st_size == 0:
                    self._fh.write(b"\0")
                    self._fh.flush()
                self._fh.seek(0)
                self._try_lock()
                self._locked = True
                return
            except BlockingIOError:
                if self._fh is not None:
                    try:
                        self._fh.close()
                    except Exception:
                        pass
                    self._fh = None
                if self.timeout_seconds <= 0:
                    raise TimeoutError(self._timeout_message())
                if (time.monotonic() - start) >= self.timeout_seconds:
                    raise TimeoutError(self._timeout_message())
                time.sleep(self.poll_seconds)
            except Exception:
                if self._fh is not None:
                    try:
                        self._fh.close()
                    except Exception:
                        pass
                    self._fh = None
                raise

    def release(self) -> None:
        if not self._locked or self._fh is None:
            return
        try:
            self._unlock()
        finally:
            try:
                self._fh.close()
            finally:
                self._fh = None
                self._locked = False

    def __enter__(self) -> "FileLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    def _try_lock(self) -> None:
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise BlockingIOError from exc
        else:
            import fcntl

            try:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise BlockingIOError from exc

    def _unlock(self) -> None:
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)

    def _timeout_message(self) -> str:
        return (
            f"DB write lock is held by another process: {self.path}. "
            "Retry later or set --lock-timeout-seconds / KEIBA_DB_LOCK_PATH."
        )


def default_lock_dir() -> Path:
    env_dir = os.environ.get("KEIBA_LOCK_DIR")
    if env_dir:
        path = Path(env_dir)
    else:
        path = Path(tempfile.gettempdir()) / "keiba_locks"
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_db_lock_path() -> Path:
    env_path = os.environ.get("KEIBA_DB_LOCK_PATH")
    if env_path:
        return Path(env_path)
    return default_lock_dir() / "db_write.lock"


@contextmanager
def db_write_lock(timeout_seconds: float = 0, poll_seconds: float = 0.25):
    lock = FileLock(default_db_lock_path(), timeout_seconds=timeout_seconds, poll_seconds=poll_seconds)
    lock.acquire()
    try:
        yield lock
    finally:
        lock.release()
