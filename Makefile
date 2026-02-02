PY ?= py64_analysis/.venv/bin/python

.PHONY: verify pdf-spec ci

ifeq ($(OS),Windows_NT)
ci:
	powershell -ExecutionPolicy Bypass -File scripts/ci.ps1
else
ci:
	bash scripts/ci.sh
endif

verify: ci

pdf-spec:
	@if [ -z "$(PDF)" ]; then \
		echo "Usage: make pdf-spec PDF=path/to/file.pdf [METHOD=pymupdf|pdfplumber|pdftotext] [OUT_DIR=docs/specs]"; \
		exit 2; \
	fi
	$(PY) tools/pdf/pdf_to_md.py $(PDF) $(if $(METHOD),--method $(METHOD),) $(if $(OUT_DIR),--output-dir $(OUT_DIR),)
