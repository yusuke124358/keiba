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


def test_ensure_no_placeholders_allows_comparisons_across_lines():
    # Previously: "<=" and ">" on different lines could be incorrectly treated
    # as a "<...>" placeholder and cause false failures for "accept" decisions.
    text = "- p_one_sided (P(delta<=0)): 0.0\n- rationale: delta_ROI_CI95.lower > 0\n"
    MODULE.ensure_no_placeholders(text)


def test_ensure_no_placeholders_detects_template_tokens():
    with pytest.raises(RuntimeError):
        MODULE.ensure_no_placeholders("ROI: <value>\n")

