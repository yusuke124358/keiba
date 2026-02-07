from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[2]
    module_path = root / "scripts" / "agent" / "run_experiment.py"
    spec = importlib.util.spec_from_file_location("run_experiment", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()


def test_coerce_eval_command_none():
    assert MODULE.coerce_eval_command(None) == []


def test_coerce_eval_command_string():
    assert MODULE.coerce_eval_command("echo hi") == ["echo hi"]


def test_coerce_eval_command_tokens_joined():
    cmd = [
        "py64_analysis/scripts/run_holdout.py",
        "--train-start",
        "2020-01-01",
        "--test-start",
        "2024-01-01",
        "--test-end",
        "2024-12-31",
    ]
    out = MODULE.coerce_eval_command(cmd)
    assert len(out) == 1
    assert "run_holdout.py" in out[0]
    assert "--train-start 2020-01-01" in out[0]


def test_select_holdout_tokens_defaults_when_missing():
    run_id = "exp_20260207_000000"
    tokens = MODULE.select_holdout_tokens([], run_id)
    assert tokens
    assert tokens[0].endswith("run_holdout.py")
    assert "--test-start" in tokens
    assert "--test-end" in tokens
    assert "--name" in tokens


def test_select_holdout_tokens_strips_python_prefix():
    run_id = "exp_20260207_000000"
    eval_cmds = [
        f"python py64_analysis/scripts/run_holdout.py --test-start 2024-01-01 --test-end 2024-12-31 --name {run_id}"
    ]
    tokens = MODULE.select_holdout_tokens(eval_cmds, run_id)
    assert tokens[0].endswith("run_holdout.py")
    assert tokens[0] != "python"


def test_parse_arg_space_and_equals():
    tokens = ["--foo", "bar", "--baz=qux"]
    assert MODULE._parse_arg(tokens, "--foo") == "bar"
    assert MODULE._parse_arg(tokens, "--baz") == "qux"

