param(
  [string]$BaseBranch = "main",
  [string]$PrTitle = "",
  [string]$PrBodyFile = "",
  [string]$Labels = ""
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
$env:GIT_TERMINAL_PROMPT = "0"
if ($LASTEXITCODE -eq 0) {
  & git -c credential.helper= -c credential.helper="!gh auth git-credential" push
} else {
  & git -c credential.helper= -c credential.helper="!gh auth git-credential" push -u origin $branch
}

if (-not $PrTitle) { $PrTitle = $branch }

$prExists = $false
& gh pr view $branch | Out-Null
if ($LASTEXITCODE -eq 0) { $prExists = $true }

if (-not $prExists) {
  if ($PrBodyFile -and (Test-Path $PrBodyFile)) {
    & gh pr create --base $BaseBranch --head $branch --title $PrTitle --body-file $PrBodyFile
  } else {
    & gh pr create --base $BaseBranch --head $branch --title $PrTitle --body "Automated publisher PR for $branch"
  }
}

if (-not $Labels) { $Labels = $env:PR_LABELS }
if ($Labels) {
  $labelList = $Labels.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" }
  foreach ($label in $labelList) {
    & gh pr edit $branch --add-label $label | Out-Null
  }
}

if ($prExists) {
  Write-Host "PR already exists for $branch."
}
