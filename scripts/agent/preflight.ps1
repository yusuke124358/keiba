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

  $hasMake = [bool](Get-Command make -ErrorAction SilentlyContinue)
  $ciPs1 = Join-Path $root "scripts\\ci.ps1"
  $verifyPs1 = Join-Path $root "scripts\\verify.ps1"
  $hasCiScript = (Test-Path $ciPs1) -or (Test-Path $verifyPs1)

  if (-not $hasMake -and -not $hasCiScript) {
    Write-Error "make not found and no scripts/ci.ps1 or scripts/verify.ps1 present."
    exit 2
  }

  if ($hasMake) {
    & make -n ci | Out-Null
    if ($LASTEXITCODE -ne 0) {
      Write-Error "make ci target missing. Add it to Makefile (verify wrappers should call make ci)."
      exit 2
    }
  } else {
    Write-Host "make not found; using PowerShell CI script fallback."
  }

  Write-Host "Preflight OK."
} finally {
  Pop-Location
}
