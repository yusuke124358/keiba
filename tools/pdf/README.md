# PDF Tools

This folder provides PDF-to-text and PDF-to-Markdown conversion utilities.

## Requirements
Install one of the following (preferred order):
- PyMuPDF (`pip install pymupdf`)
- pdfplumber (`pip install pdfplumber`)
- pdftotext (Poppler) on PATH

## Usage
Convert a PDF to Markdown and meta JSON:
```bash
python tools/pdf/pdf_to_md.py "docs/_private/jravan_spec.pdf"
```

Specify extractor method and output directory:
```bash
python tools/pdf/pdf_to_md.py --method pymupdf --output-dir docs/specs "docs/_private/jravan_spec.pdf"
```

Extract text only:
```bash
python tools/pdf/pdf_to_text.py "docs/_private/jravan_spec.pdf" > /tmp/spec.txt
```

Make target:
```bash
make pdf-spec PDF=docs/_private/jravan_spec.pdf
```

Outputs:
- `docs/specs/<basename>.md`
- `docs/specs/<basename>.meta.json`
