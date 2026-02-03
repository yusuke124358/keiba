#!/usr/bin/env python3
import argparse
from pathlib import Path
import subprocess
import sys


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> int:
    p = argparse.ArgumentParser(description="Ingest a PDF into docs/knowledge.")
    p.add_argument("pdf", help="Path to the PDF file")
    p.add_argument(
        "--kind",
        choices=["jra_van", "papers", "misc"],
        default="papers",
        help="Knowledge subdirectory under docs/knowledge",
    )
    p.add_argument(
        "--method",
        choices=["pymupdf", "pdfplumber", "pdftotext"],
        default=None,
        help="Extraction method",
    )
    args = p.parse_args()

    root = repo_root()
    pdf_path = Path(args.pdf).expanduser().resolve()
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    out_dir = root / "docs" / "knowledge" / args.kind
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [sys.executable, str(root / "tools" / "pdf" / "pdf_to_md.py"), str(pdf_path)]
    if args.method:
        cmd += ["--method", args.method]
    cmd += ["--output-dir", str(out_dir)]

    result = subprocess.run(cmd, cwd=root)
    if result.returncode != 0:
        raise SystemExit(result.returncode)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
