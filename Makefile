PY ?= py64_analysis/.venv/bin/python

.PHONY: verify pdf-spec ci

ci:
	bash scripts/ci.sh

verify: ci

pdf-spec:
	@if [ -z "$(PDF)" ]; then \
		echo "Usage: make pdf-spec PDF=path/to/file.pdf [METHOD=pymupdf|pdfplumber|pdftotext] [OUT_DIR=docs/specs]"; \
		exit 2; \
	fi
	$(PY) tools/pdf/pdf_to_md.py $(PDF) $(if $(METHOD),--method $(METHOD),) $(if $(OUT_DIR),--output-dir $(OUT_DIR),)
