#!/usr/bin/env python3
import re
import sys
from pathlib import Path

import yaml

REQUIRED_KEYS = {
    "id",
    "source",
    "how_to_get",
    "license",
    "date_range",
    "refresh",
    "checksum",
    "schema_ref",
    "notes",
}

CHECKSUM_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
ALLOWED_CHECKSUM = {"sha256:pending", "sha256:unknown"}
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\s+to\s+\d{4}-\d{2}-\d{2}$")
ID_RE = re.compile(r"^[a-z0-9_-]+$")


def main():
    manifest_path = Path("data/manifest.yml")
    if not manifest_path.exists():
        print("Missing data/manifest.yml", file=sys.stderr)
        return 1

    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    errors = []

    if not isinstance(data, dict):
        errors.append("Manifest must be a mapping")
        data = {}

    version = data.get("version")
    if not isinstance(version, int):
        errors.append("version must be an integer")

    datasets = data.get("datasets")
    if not isinstance(datasets, list):
        errors.append("datasets must be a list")
        datasets = []

    seen_ids = set()
    for idx, entry in enumerate(datasets):
        if not isinstance(entry, dict):
            errors.append(f"datasets[{idx}] must be a mapping")
            continue

        missing = REQUIRED_KEYS - set(entry.keys())
        for key in sorted(missing):
            errors.append(f"datasets[{idx}] missing key: {key}")

        dataset_id = entry.get("id", "")
        if not dataset_id or not ID_RE.match(dataset_id):
            errors.append(f"datasets[{idx}] id invalid: {dataset_id}")
        if dataset_id in seen_ids:
            errors.append(f"Duplicate dataset id: {dataset_id}")
        seen_ids.add(dataset_id)

        checksum = entry.get("checksum", "")
        if checksum not in ALLOWED_CHECKSUM and not CHECKSUM_RE.match(checksum):
            errors.append(f"datasets[{idx}] checksum invalid: {checksum}")

        date_range = entry.get("date_range", "")
        if date_range.lower() != "n/a" and not DATE_RE.match(date_range):
            errors.append(f"datasets[{idx}] date_range invalid: {date_range}")

        schema_ref = entry.get("schema_ref", "")
        if schema_ref and not Path(schema_ref).exists():
            errors.append(f"datasets[{idx}] schema_ref not found: {schema_ref}")

    if errors:
        print("Manifest validation failed:", file=sys.stderr)
        for err in errors:
            print(f"- {err}", file=sys.stderr)
        return 1

    print("Manifest validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
