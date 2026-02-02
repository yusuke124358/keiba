#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


TYPE_MAP = {
    "int": "integer",
    "integer": "integer",
    "float": "number",
    "double": "number",
    "number": "number",
    "bool": "boolean",
    "boolean": "boolean",
}


def _normalize_type(value: str):
    if not value:
        return "string"
    v = value.lower()
    for key, mapped in TYPE_MAP.items():
        if key in v:
            return mapped
    return "string"


def _parse_tables(text: str):
    lines = text.splitlines()
    rows = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("|") and "|" in line[1:]:
            header = [c.strip() for c in line.strip("|").split("|")]
            if i + 1 < len(lines) and set(lines[i + 1].strip()) <= {"|", "-", ":", " "}:
                i += 2
                while i < len(lines) and lines[i].strip().startswith("|"):
                    row = [c.strip() for c in lines[i].strip("|").split("|")]
                    rows.append((header, row))
                    i += 1
                continue
        i += 1
    return rows


def _extract_fields(text: str):
    fields = []
    for header, row in _parse_tables(text):
        header_lower = [h.lower() for h in header]
        try:
            name_idx = next(
                i
                for i, h in enumerate(header_lower)
                if "field" in h or "name" in h or "項目" in h
            )
        except StopIteration:
            name_idx = 0
        try:
            type_idx = next(
                i for i, h in enumerate(header_lower) if "type" in h or "型" in h
            )
        except StopIteration:
            type_idx = None
        try:
            desc_idx = next(
                i for i, h in enumerate(header_lower) if "desc" in h or "説明" in h
            )
        except StopIteration:
            desc_idx = None

        if name_idx >= len(row):
            continue
        name = row[name_idx].strip()
        if not name or name.lower() == "name":
            continue
        field_type = (
            _normalize_type(row[type_idx])
            if type_idx is not None and type_idx < len(row)
            else "string"
        )
        desc = row[desc_idx] if desc_idx is not None and desc_idx < len(row) else ""
        fields.append({"name": name, "type": field_type, "description": desc})
    return fields


def _record_name(path: Path):
    stem = path.stem
    if stem.startswith("jravan_"):
        return stem[len("jravan_") :]
    return stem


def main():
    parser = argparse.ArgumentParser(
        description="Generate JSON schema templates from JRA-VAN spec markdown"
    )
    parser.add_argument(
        "--input", action="append", help="Input markdown file (can repeat)"
    )
    parser.add_argument(
        "--glob",
        default="docs/specs/jravan_*.md",
        help="Glob pattern when --input not supplied",
    )
    parser.add_argument("--out-dir", default="schemas/jravan")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    inputs = [Path(p) for p in (args.input or [])]
    if not inputs:
        inputs = [Path(p) for p in Path().glob(args.glob)]

    if not inputs:
        raise SystemExit("No spec markdown files found.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in inputs:
        text = path.read_text(encoding="utf-8")
        fields = _extract_fields(text)
        record = _record_name(path)
        schema_path = out_dir / f"{record}.schema.json"

        if schema_path.exists() and not args.force:
            raise SystemExit(f"Schema exists: {schema_path} (use --force to overwrite)")

        schema = {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": f"JRA-VAN {record}",
            "type": "object",
            "properties": {},
            "additionalProperties": True,
            "$comment": "Generated from spec markdown; review and refine types/required fields.",
        }

        for field in fields:
            schema["properties"][field["name"]] = {
                "type": field["type"],
                "description": field["description"],
            }

        schema_path.write_text(
            json.dumps(schema, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Wrote: {schema_path}")


if __name__ == "__main__":
    main()
