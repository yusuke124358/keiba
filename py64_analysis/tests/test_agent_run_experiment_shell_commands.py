from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _load_module():
    root = Path(__file__).resolve().parents[2]
    module_path = root / "scripts" / "agent" / "run_experiment.py"
    spec = importlib.util.spec_from_file_location("run_experiment", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


class _Result:
    def __init__(self, returncode: int):
        self.returncode = returncode


def test_run_shell_commands_allows_compare_failure(monkeypatch, tmp_path: Path) -> None:
    def fake_run(cmd, cwd=None, check=False, shell=False):
        if isinstance(cmd, str) and "compare_metrics_json.py" in cmd:
            return _Result(1)
        return _Result(0)

    monkeypatch.setattr(MODULE.subprocess, "run", fake_run)
    MODULE.run_shell_commands(
        [
            "python py64_analysis/scripts/compare_metrics_json.py --baseline b --candidate c",
        ],
        cwd=tmp_path,
    )


def test_run_shell_commands_raises_on_other_failures(monkeypatch, tmp_path: Path) -> None:
    def fake_run(cmd, cwd=None, check=False, shell=False):
        return _Result(1)

    monkeypatch.setattr(MODULE.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError):
        MODULE.run_shell_commands(['python -c "exit(1)"'], cwd=tmp_path)
