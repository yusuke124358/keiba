#!/usr/bin/env bash
set -euo pipefail

root=$(git rev-parse --show-toplevel)
cd "$root"

if ! command -v codex >/dev/null 2>&1; then
  echo "codex CLI not found. Install with: npm i -g @openai/codex" >&2
  exit 2
fi

if ! codex login status >/dev/null 2>&1; then
  echo "codex login status failed. Run: codex login --device-auth on the runner." >&2
  exit 2
fi

if ! command -v gh >/dev/null 2>&1; then
  echo "gh CLI not found. Install GitHub CLI and authenticate for API access." >&2
  exit 2
fi

if ! command -v make >/dev/null 2>&1; then
  echo "make not found. Install build tools or provide make ci entrypoint." >&2
  exit 2
fi

if ! make -n ci >/dev/null 2>&1; then
  echo "make ci target missing. Add it to Makefile (verify wrappers should call make ci)." >&2
  exit 2
fi

echo "Preflight OK."
