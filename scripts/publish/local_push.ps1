param(
  [string]$BaseBranch = "main",
  [string]$PrTitle = "",
  [string]$PrBodyFile = ""
)

$root = (& git rev-parse --show-toplevel).Trim()
Set-Location $root

$branch = (& git rev-parse --abbrev-ref HEAD).Trim()
if ($branch -eq "main") {
  Write-Error "Refusing to publish from main. Checkout a feature branch."
  exit 2
}

$gh = Get-Command gh -ErrorAction SilentlyContinue
if (-not $gh) {
  Write-Error "gh CLI not found. Install GitHub CLI and authenticate before publishing."
  exit 2
}

& gh auth status | Out-Null
if ($LASTEXITCODE -ne 0) {
  Write-Error "gh auth status failed. Run: gh auth login"
  exit 2
}

& git rev-parse --abbrev-ref --symbolic-full-name '@{u}' | Out-Null
if ($LASTEXITCODE -eq 0) {
  & git -c credential.helper= -c credential.helper="!gh auth git-credential" push
} else {
  & git -c credential.helper= -c credential.helper="!gh auth git-credential" push -u origin $branch
}

if (-not $PrTitle) { $PrTitle = $branch }

& gh pr view $branch | Out-Null
if ($LASTEXITCODE -eq 0) {
  Write-Host "PR already exists for $branch."
  exit 0
}

if ($PrBodyFile -and (Test-Path $PrBodyFile)) {
  & gh pr create --base $BaseBranch --head $branch --title $PrTitle --body-file $PrBodyFile
} else {
  & gh pr create --base $BaseBranch --head $branch --title $PrTitle --body "Automated publisher PR for $branch"
}
