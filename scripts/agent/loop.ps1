param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$Args
)

$root = (& git rev-parse --show-toplevel).Trim()
$py = Join-Path $root "py64_analysis\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) {
  $py = "python"
}

& $py (Join-Path $root "scripts\agent\loop.py") @Args
