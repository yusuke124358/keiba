---
name: reviewer-triage
description: Codex skill reviewer-triage
version: 0.1
---

# reviewer-triage

Purpose: normalize raw review signals into a deduplicated issues list.

Input:
- PR comments + PR reviews + CI/check summaries (JSON bundle).

Rules:
- Deduplicate by root cause (same file/line/message).
- Fill file/line when reasonably inferable.
- Prefer concise, actionable messages.
- Output must conform to `schemas/agent/review_items.schema.json`.
- Use issue id rule: `iss_` + sha1(`source|file|line|normalized_message`)[:12].

Output:
- `review_items.json` with `issues[]`.
