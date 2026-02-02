#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path


def _extract_pymupdf(path: Path):
    import fitz

    doc = fitz.open(str(path))
    pages = [page.get_text("text") for page in doc]
    meta = {
        "extractor": "pymupdf",
        "page_count": doc.page_count,
    }
    return pages, meta


def _extract_pdfplumber(path: Path):
    import pdfplumber

    pages = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
        meta = {
            "extractor": "pdfplumber",
            "page_count": len(pdf.pages),
        }
    return pages, meta


def _extract_pdftotext(path: Path):
    if not shutil.which("pdftotext"):
        raise RuntimeError("pdftotext not found in PATH")
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "out.txt"
        cmd = ["pdftotext", "-layout", str(path), str(out_path)]
        subprocess.run(cmd, check=True)
        text = out_path.read_text(encoding="utf-8", errors="ignore")
    pages = [page for page in text.split("\f")]
    meta = {
        "extractor": "pdftotext",
        "page_count": len(pages),
    }
    return pages, meta


def extract_pages(path: Path, method: str = "auto"):
    if method == "auto":
        for attempt in ("pymupdf", "pdfplumber", "pdftotext"):
            try:
                return extract_pages(path, method=attempt)
            except Exception:
                continue
        raise RuntimeError("No PDF extractor available. Install PyMuPDF or pdfplumber, or add pdftotext.")

    if method == "pymupdf":
        return _extract_pymupdf(path)
    if method == "pdfplumber":
        return _extract_pdfplumber(path)
    if method == "pdftotext":
        return _extract_pdftotext(path)

    raise ValueError(f"Unknown method: {method}")


def main():
    parser = argparse.ArgumentParser(description="Extract text from a PDF file")
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument("--method", default="auto", choices=["auto", "pymupdf", "pdfplumber", "pdftotext"])
    parser.add_argument("--output", help="Optional output text file")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise SystemExit(f"PDF not found: {pdf_path}")

    pages, meta = extract_pages(pdf_path, method=args.method)
    meta["source"] = str(pdf_path)
    meta["timestamp_utc"] = datetime.utcnow().isoformat() + "Z"

    text = "\n\n".join(page.rstrip() for page in pages).strip() + "\n"

    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text, end="")

    print(json.dumps(meta, ensure_ascii=True, indent=2), file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
