Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = (& git rev-parse --show-toplevel).Trim()
$py = Join-Path $root "py64_analysis\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

$baseRef = $env:VERIFY_BASE
if (-not $baseRef) { $baseRef = "origin/main" }

function Run-Step([string]$name, [scriptblock]$block) {
  & $block
  if ($LASTEXITCODE -ne 0) {
    throw "$name failed with exit code $LASTEXITCODE"
  }
}

Run-Step "ruff format" { & $py -m ruff format --check py64_analysis }
Run-Step "ruff check" { & $py -m ruff check py64_analysis }
Run-Step "mypy" { & $py -m mypy py64_analysis\src }
Run-Step "pytest" { & $py -m pytest py64_analysis\tests }
Run-Step "check_system_status" { & $py py64_analysis\scripts\check_system_status.py }
Run-Step "validate_data_manifest" { & $py scripts\validate_data_manifest.py }
Run-Step "validate_json_schemas" { & $py scripts\validate_json_schemas.py }
Run-Step "verify_experiment_log" { & $py scripts\verify_experiment_log.py --base $baseRef }

if ($env:PDF_SPEC) {
  $args = @($env:PDF_SPEC)
  if ($env:PDF_SPEC_METHOD) { $args = @('--method', $env:PDF_SPEC_METHOD) + $args }
  if ($env:PDF_SPEC_OUT_DIR) { $args = @('--output-dir', $env:PDF_SPEC_OUT_DIR) + $args }
  Run-Step "pdf-spec" { & $py tools\pdf\pdf_to_md.py @args }
}
