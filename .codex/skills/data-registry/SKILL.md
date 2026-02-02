---
name: data-registry
description: Maintain data/manifest.yml, schemas, and dictionaries with validation when data changes.
---

# Data Registry

## Steps
1) Update `data/manifest.yml` for any dataset additions/changes.
2) Update schema files under `schemas/` and dictionaries under `docs/data_dictionary/`.
3) Run validation:
   - `python scripts/validate_data_manifest.py`
   - `python scripts/validate_json_schemas.py`
4) Record changes in experiment logs if code changes are involved.

## Notes
- Keep checksum and schema_ref aligned with actual assets.
- Use `sha256:pending` only when a checksum is not yet computed.
