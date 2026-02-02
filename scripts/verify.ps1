Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = (& git rev-parse --show-toplevel).Trim()
Push-Location $root
try {
  & make ci
  if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
  }
} finally {
  Pop-Location
}
