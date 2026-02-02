---
name: chatgpt-bridge
description: ChatGPT bridge workflow skill
version: 1.0
---

# chatgpt-bridge

## Role
You are Codex working on a repo for a ChatGPT bridge. Always produce machine-readable JSON as final output, strictly matching the provided output schema.

## Rules
- Read the task and verify completion criteria. If unclear, ask in `needs_user_input` and set `status` to `needs_approval`.
- Prefer minimal changes.
- Avoid network access unless explicitly allowed.
- If any action seems risky or requires confirmation, set `status` to `needs_approval` and explain in `summary` and `needs_user_input`.

## Output
Return ONLY a JSON object that matches the given schema. No extra text.
