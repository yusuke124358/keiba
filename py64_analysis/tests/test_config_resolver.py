import shutil
from pathlib import Path

import yaml

from keiba.utils.config_resolver import resolve_config_path, save_config_used


def _make_tmp_dir() -> Path:
    base = Path(__file__).resolve().parent / "_tmp_config_resolver"
    if base.exists():
        shutil.rmtree(base, ignore_errors=True)
    base.mkdir(parents=True, exist_ok=True)
    return base


def test_resolve_config_path_priority(monkeypatch) -> None:
    base = _make_tmp_dir()
    try:
        env_cfg = base / "env.yaml"
        cli_cfg = base / "cli.yaml"
        env_cfg.write_text("a: 1\n", encoding="utf-8")
        cli_cfg.write_text("b: 2\n", encoding="utf-8")

        monkeypatch.setenv("KEIBA_CONFIG_PATH", str(env_cfg))
        path, origin = resolve_config_path(str(cli_cfg))
        assert origin == "env:KEIBA_CONFIG_PATH"
        assert path == env_cfg.resolve()

        monkeypatch.delenv("KEIBA_CONFIG_PATH", raising=False)
        path, origin = resolve_config_path(str(cli_cfg))
        assert origin == "cli:--config"
        assert path == cli_cfg.resolve()

        path, origin = resolve_config_path(None)
        assert origin.startswith("default:")
        assert path.name == "config.yaml"
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_save_config_used() -> None:
    base = _make_tmp_dir()
    try:
        cfg = base / "config.yaml"
        cfg.write_text(yaml.safe_dump({"foo": 1, "bar": 2}, sort_keys=True), encoding="utf-8")
        run_dir = base / "run"
        run_dir.mkdir()

        meta = save_config_used(cfg, run_dir)
        out_path = run_dir / "config_used.yaml"
        assert out_path.exists()
        assert meta["config_hash_sha256"]
        assert len(meta["config_hash_sha256"]) == 64
    finally:
        shutil.rmtree(base, ignore_errors=True)
