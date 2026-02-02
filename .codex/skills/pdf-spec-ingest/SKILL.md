---
name: pdf-spec-ingest
description: Codex skill pdf-spec-ingest
version: 0.1
---

﻿---
name: pdf-spec-ingest
description: Ingest a PDF spec into docs/specs, extract key fields, and update the JRA-VAN data dictionary and validations.
---

# PDF Spec Ingest

## Steps
1) Convert the PDF into Markdown assets:
   - `python tools/pdf/pdf_to_md.py <pdf_path>`
2) Read the generated `docs/specs/<basename>.md` and summarize:
   - Important sections/constraints
   - Data fields (name/type/description)
   - Warnings or prohibited usage
3) Update `docs/data_dictionary/jravan.yml` with new/updated fields.
4) If new fields affect schemas or validation, update `schemas/jravan/*.schema.json` and add tests if needed.

## Notes
- Store raw PDFs under `docs/_private/` (gitignored). Only commit extracted assets.
- Keep changes small and scoped to the spec being ingested.
