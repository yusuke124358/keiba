from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Optional

import yaml

from ..config import PROJECT_ROOT


def resolve_config_path(cli_config: Optional[str]) -> tuple[Path, str]:
    env_path = os.environ.get("KEIBA_CONFIG_PATH")
    if env_path:
        return _normalize_path(Path(env_path)), "env:KEIBA_CONFIG_PATH"
    if cli_config:
        return _normalize_path(Path(cli_config)), "cli:--config"
    default_path = Path("config") / "config.yaml"
    return _normalize_path(default_path), "default:config/config.yaml"


def load_yaml(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data or {}


def dump_yaml_canonical(data: dict, path: Path) -> None:
    text = yaml.safe_dump(data, sort_keys=True, allow_unicode=True)
    path.write_text(text, encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def save_config_used(resolved_config_path: Path, run_dir: Path) -> dict:
    if not run_dir.exists():
        raise SystemExit(f"run_dir not found: {run_dir}")
    data = load_yaml(resolved_config_path)
    out_path = run_dir / "config_used.yaml"
    dump_yaml_canonical(data, out_path)
    return {
        "config_used_path": str(out_path),
        "config_hash_sha256": sha256_file(out_path),
    }


def save_config_origin(run_dir: Path, origin_payload: dict) -> None:
    out_path = run_dir / "config_origin.json"
    out_path.write_text(json.dumps(origin_payload, ensure_ascii=False, indent=2), encoding="utf-8")


def git_commit(project_root: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(project_root))
        return out.decode("utf-8").strip()
    except Exception:
        return None


def rel_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except Exception:
        return str(path)


def _normalize_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    if PROJECT_ROOT:
        return (PROJECT_ROOT / path).resolve()
    return path.resolve()
