---
name: paper-ingest
description: Codex skill paper-ingest
version: 0.1
---

﻿---
name: paper-ingest
description: Convert a research paper PDF into a structured markdown summary with reproducible details.
---

# Paper Ingest

## Steps
1) Convert the PDF into Markdown assets:
   - `python tools/pdf/pdf_to_md.py <pdf_path>`
2) Create `docs/research/papers/<slug>.md` using `docs/research/papers/_template.md`.
3) Extract and record:
   - Features, model, training details
   - Data splits and leakage considerations
   - Evaluation metrics and baselines
4) Add links and license notes.

## Notes
- Keep reproduction hints concise and actionable.
- Cite URLs when available.
