Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = (& git rev-parse --show-toplevel).Trim()
Push-Location $root
try {
  if (-not (Get-Command codex -ErrorAction SilentlyContinue)) {
    Write-Error "codex CLI not found. Install with: npm i -g @openai/codex"
    exit 2
  }

  & codex login status | Out-Null
  if ($LASTEXITCODE -ne 0) {
    Write-Error "codex login status failed. Run: codex login --device-auth on the runner."
    exit 2
  }

  if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    Write-Error "gh CLI not found. Install GitHub CLI and authenticate for API access."
    exit 2
  }

  if (-not (Get-Command make -ErrorAction SilentlyContinue)) {
    Write-Error "make not found. Install build tools or provide make ci entrypoint."
    exit 2
  }

  & make -n ci | Out-Null
  if ($LASTEXITCODE -ne 0) {
    Write-Error "make ci target missing. Add it to Makefile (verify wrappers should call make ci)."
    exit 2
  }

  Write-Host "Preflight OK."
} finally {
  Pop-Location
}
