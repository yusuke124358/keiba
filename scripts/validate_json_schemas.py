#!/usr/bin/env python3
import json
import sys
from pathlib import Path


def main():
    schema_root = Path("schemas")
    if not schema_root.exists():
        print("No schemas/ directory found. Skipping.")
        return 0

    schema_files = list(schema_root.rglob("*.schema.json"))
    if not schema_files:
        print("No schema files found under schemas/. Skipping.")
        return 0

    errors = []
    for path in schema_files:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"{path}: invalid JSON ({exc})")
            continue

        if "$schema" not in data:
            errors.append(f"{path}: missing $schema")
        if "type" not in data:
            errors.append(f"{path}: missing type")

    if errors:
        print("Schema validation failed:", file=sys.stderr)
        for err in errors:
            print(f"- {err}", file=sys.stderr)
        return 1

    print("Schema validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
