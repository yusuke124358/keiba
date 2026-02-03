import textwrap
from pathlib import Path

import pytest


def _write(tmp_path: Path, content: str) -> Path:
    path = tmp_path / "log.md"
    path.write_text(textwrap.dedent(content).strip() + "\n", encoding="utf-8")
    return path


def test_eval_gate_infra_allows_na(tmp_path):
    from scripts.eval_gate import validate_eval_fields

    path = _write(
        tmp_path,
        """
        - Experiment type: infra
        - ROI: N/A
        - Total stake: N/A
        - n_bets: N/A
        - Test period: N/A
        - Max drawdown: N/A
        """,
    )
    assert validate_eval_fields(path) == []


def test_eval_gate_experiment_requires_metrics(tmp_path):
    from scripts.eval_gate import validate_eval_fields

    path = _write(
        tmp_path,
        """
        - Experiment type: experiment
        - ROI: N/A
        - Total stake: N/A
        - n_bets: N/A
        - Test period: 2020-01-01 to 2024-12-31
        - Max drawdown: N/A
        """,
    )
    errors = validate_eval_fields(path)
    assert any("roi must not be N/A" in e.lower() for e in errors)
    assert any("total stake must not be N/A" in e.lower() for e in errors)
    assert any("n_bets must not be N/A" in e.lower() for e in errors)
    assert any("max drawdown must not be N/A" in e.lower() for e in errors)


def test_eval_gate_experiment_test_period_format(tmp_path):
    from scripts.eval_gate import validate_eval_fields

    path = _write(
        tmp_path,
        """
        - Experiment type: experiment
        - ROI: 0.01
        - Total stake: 100
        - n_bets: 10
        - Test period: 2020/01/01-2024/12/31
        - Max drawdown: 0.2
        """,
    )
    errors = validate_eval_fields(path)
    assert any("test period" in e.lower() for e in errors)
