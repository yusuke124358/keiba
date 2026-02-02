#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from pdf_to_text import extract_pages


def _slug_base(path: Path):
    return path.stem


def _hash_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_table_row(line: str):
    if not line.strip():
        return False, []
    cols = re.split(r"\s{2,}", line.strip())
    cols = [c.strip() for c in cols if c.strip()]
    if len(cols) >= 3:
        return True, cols
    return False, []


def _emit_table(rows):
    if not rows:
        return []
    header = rows[0]
    width = len(header)
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * width) + " |"]
    for row in rows[1:]:
        padded = row + [""] * (width - len(row))
        out.append("| " + " | ".join(padded[:width]) + " |")
    out.append("")
    return out


def pages_to_markdown(pages):
    out_lines = []
    for idx, page in enumerate(pages, start=1):
        out_lines.append(f"\n\n## Page {idx}\n")
        table_rows = []
        for raw_line in page.splitlines():
            line = raw_line.rstrip()
            is_table, cols = _is_table_row(line)
            if is_table:
                table_rows.append(cols)
                continue
            if table_rows:
                out_lines.extend(_emit_table(table_rows))
                table_rows = []
            out_lines.append(line)
        if table_rows:
            out_lines.extend(_emit_table(table_rows))
    return "\n".join(out_lines).strip() + "\n"


def main():
    parser = argparse.ArgumentParser(description="Convert PDF to Markdown")
    parser.add_argument("pdf", help="Path to PDF")
    parser.add_argument(
        "--method",
        default="auto",
        choices=["auto", "pymupdf", "pdfplumber", "pdftotext"],
    )
    parser.add_argument("--output-dir", default="docs/specs")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base = _slug_base(pdf_path)
    md_path = output_dir / f"{base}.md"
    meta_path = output_dir / f"{base}.meta.json"

    if (md_path.exists() or meta_path.exists()) and not args.force:
        raise SystemExit("Output exists. Use --force to overwrite.")

    pages, meta = extract_pages(pdf_path, method=args.method)
    markdown = pages_to_markdown(pages)

    md_path.write_text(markdown, encoding="utf-8")

    meta.update(
        {
            "source": str(pdf_path),
            "output_md": str(md_path),
            "output_meta": str(meta_path),
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "sha256": _hash_file(pdf_path),
            "file_size_bytes": pdf_path.stat().st_size,
        }
    )

    meta_path.write_text(
        json.dumps(meta, ensure_ascii=True, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Wrote: {md_path}")
    print(f"Meta: {meta_path}")


if __name__ == "__main__":
    sys.exit(main())
