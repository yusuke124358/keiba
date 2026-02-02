#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


def normalize_text(text: str) -> str:
    lines = text.splitlines()
    return "\n".join(lines).rstrip("\n") + "\n"


def write_text_lf(path: Path, text: str) -> None:
    normalized = normalize_text(text)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(normalized)


def ensure_frontmatter(path: Path) -> None:
    raw = path.read_text(encoding="utf-8-sig")
    stripped = raw.lstrip("\ufeff\r\n\t ")
    if not stripped.startswith("---"):
        name = path.parent.name
        fm = (
            f"---\nname: {name}\ndescription: Codex skill {name}\nversion: 0.1\n---\n\n"
        )
        stripped = fm + stripped
    write_text_lf(path, stripped)


def normalize_schema(path: Path) -> None:
    raw = path.read_text(encoding="utf-8-sig")
    json.loads(raw)
    write_text_lf(path, raw)


def main() -> int:
    root = Path.cwd()
    skills_root = root / ".codex" / "skills"
    if skills_root.exists():
        for skill in skills_root.rglob("SKILL.md"):
            ensure_frontmatter(skill)

    schema_path = root / "scripts" / "agent" / "output_schema.json"
    if schema_path.exists():
        normalize_schema(schema_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
