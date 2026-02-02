---
name: jravan-spec-to-schema
description: Codex skill jravan-spec-to-schema
version: 0.1
---

﻿---
name: jravan-spec-to-schema
description: Generate or update JRA-VAN JSON schemas from spec markdown and validate outputs.
---

# JRA-VAN Spec to Schema

## Steps
1) Ensure `docs/specs/jravan_*.md` exists (use pdf-spec-ingest first).
2) Run schema generator:
   - `python tools/jravan/spec_to_schema.py --glob "docs/specs/jravan_*.md"`
3) Review schema diffs and refine types/required fields as needed.
4) Validate schemas:
   - `python scripts/validate_json_schemas.py`

## Notes
- Generated schemas are templates; manual review is required.
