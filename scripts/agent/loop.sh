#!/usr/bin/env bash
set -euo pipefail
root=$(git rev-parse --show-toplevel)
py="$root/py64_analysis/.venv/bin/python"
if [ ! -x "$py" ]; then
  py="python3"
fi
"$py" "$root/scripts/agent/loop.py" "$@"
