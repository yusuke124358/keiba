from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

from keiba.stats.block_bootstrap import BootstrapSettings, compute_block_bootstrap_summary


def _load_cli_module():
    root = Path(__file__).resolve().parents[2]
    module_path = root / "scripts" / "stats" / "block_bootstrap.py"
    spec = importlib.util.spec_from_file_location("block_bootstrap_cli", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_block_bootstrap_cli_writes_expected_json(tmp_path, monkeypatch):
    variant = tmp_path / "variant.csv"
    baseline = tmp_path / "baseline.csv"
    out_path = tmp_path / "out.json"

    variant.write_text(
        "\n".join(
            [
                "date,race_id,stake,return,profit",
                "2024-01-01,2024010101010101,100,200,100",
                "2024-01-01,2024010101010102,100,0,-100",
                "2024-01-02,2024010201010101,200,0,-200",
            ]
        ),
        encoding="utf-8",
    )
    baseline.write_text(
        "\n".join(
            [
                "date,race_id,stake,return,profit",
                "2024-01-01,2024010101010101,100,0,-100",
                "2024-01-02,2024010201010101,200,400,200",
            ]
        ),
        encoding="utf-8",
    )

    B = 100
    seed = 123
    settings = BootstrapSettings(B=B, seed=seed)
    expected = compute_block_bootstrap_summary(variant, baseline, settings)

    module = _load_cli_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "block_bootstrap.py",
            "--variant",
            str(variant),
            "--baseline",
            str(baseline),
            "--out",
            str(out_path),
            "--B",
            str(B),
            "--seed",
            str(seed),
        ],
    )
    rc = module.main()
    assert rc == 0
    assert out_path.exists()

    got = json.loads(out_path.read_text(encoding="utf-8"))
    assert got == expected
