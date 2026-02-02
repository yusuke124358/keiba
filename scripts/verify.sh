#!/usr/bin/env bash
set -euo pipefail

root=$(git rev-parse --show-toplevel)
cd "$root"

py="$root/py64_analysis/.venv/bin/python"
if [ ! -x "$py" ]; then
  py="python3"
fi

base_ref="${VERIFY_BASE:-origin/main}"

"$py" -m ruff format --check py64_analysis
"$py" -m ruff check py64_analysis
"$py" -m mypy py64_analysis/src
"$py" -m pytest py64_analysis/tests
"$py" py64_analysis/scripts/check_system_status.py
"$py" scripts/validate_data_manifest.py
"$py" scripts/validate_json_schemas.py
"$py" scripts/verify_experiment_log.py --base "$base_ref"

if [ -n "${PDF_SPEC:-}" ]; then
  make pdf-spec PDF="$PDF_SPEC" ${PDF_SPEC_METHOD:+METHOD=$PDF_SPEC_METHOD} ${PDF_SPEC_OUT_DIR:+OUT_DIR=$PDF_SPEC_OUT_DIR}
fi
