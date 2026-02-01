param(
  [Parameter(Mandatory = $true)]
  [ValidateSet("guard", "reviewer", "courier")]
  [string]$Role,
  [string]$Inbox = "tasks/inbox",
  [string]$Outbox = "tasks/outbox",
  [int]$PollSeconds = 3
)

$profiles = @{
  guard = "experiment"
  reviewer = "reviewer"
  courier = "publisher"
}

function Ensure-Dir([string]$Path) {
  if (-not (Test-Path $Path)) {
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
  }
}

function Get-OwnerRole([string]$Text) {
  $lines = $Text -split "`r?`n"
  for ($i = 0; $i -lt $lines.Length; $i++) {
    $line = $lines[$i]
    if ($line -match '^\s*Owner\s*/\s*role\s*:\s*(.+)$') { return $Matches[1].Trim() }
    if ($line -match '^\s*##\s*Owner\s*/\s*role\s*$') {
      if ($i + 1 -lt $lines.Length -and $lines[$i + 1] -match '^\s*-\s*(.+)$') {
        return $Matches[1].Trim()
      }
    }
  }
  return $null
}

function Role-Matches([string]$Text, [string]$Role) {
  $owner = Get-OwnerRole $Text
  if (-not $owner) { return $false }
  switch ($Role) {
    "guard" { return $owner -match "guard|experiment" }
    "reviewer" { return $owner -match "reviewer|inspector" }
    default { return $false }
  }
}

function Get-ReportPath([string]$Text) {
  $match = [regex]::Match($Text, '(?im)^\s*-\s*(tasks[\\/]+outbox[\\/]+[^\s]+)$')
  if ($match.Success) {
    return ($match.Groups[1].Value -replace '/', '\\')
  }
  return $null
}

function Get-SectionLines([string]$Text, [string]$Header) {
  $lines = $Text -split "`r?`n"
  $collect = $false
  $out = @()
  foreach ($line in $lines) {
    if ($line -match '^\s*##\s+') {
      if ($collect) { break }
      if ($line -match ('^\s*##\s*' + [regex]::Escape($Header) + '\s*$')) {
        $collect = $true
      }
      continue
    }
    if ($collect) { $out += $line }
  }
  return $out
}

function Get-ReviewScope([string]$Text) {
  $lines = Get-SectionLines $Text 'review_scope (base/head)'
  $base = $null
  $head = $null
  foreach ($line in $lines) {
    if ($line -match '^\s*-\s*Base:\s*(.+)$') { $base = $Matches[1].Trim() }
    if ($line -match '^\s*-\s*Head:\s*(.+)$') { $head = $Matches[1].Trim() }
  }
  return @{ Base = $base; Head = $head }
}

function Get-AutoReview([string]$Text) {
  $lines = Get-SectionLines $Text 'Auto-review (optional)'
  $enable = $false
  $taskPath = $null
  $reportPath = $null
  foreach ($line in $lines) {
    if ($line -match '^\s*-\s*Enable:\s*yes\b') { $enable = $true }
    if ($line -match '^\s*-\s*Review task path:\s*(.+)$') { $taskPath = $Matches[1].Trim() }
    if ($line -match '^\s*-\s*Review report path:\s*(.+)$') { $reportPath = $Matches[1].Trim() }
  }
  return @{ Enable = $enable; TaskPath = $taskPath; ReportPath = $reportPath }
}

function Resolve-PathSafe([string]$Path, [string]$RepoRoot) {
  if (-not $Path) { return $null }
  if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
  return (Join-Path $RepoRoot $Path)
}

function New-ReviewerTaskContent([string]$TaskId, [string]$SourceTask, [string]$GuardReport, [string]$Base, [string]$Head, [string]$ReviewReportPath) {
  @"
# Task: review $TaskId

## Purpose
- Review PR diff only

## Inputs
- Source task: $SourceTask
- Guard report: $GuardReport

## review_scope (base/head)
- Base: $Base
- Head: $Head

## Constraints
- Allowed:
  - /review only
- Forbidden:
  - edits, execution, DB ops, git push

## Execution viability
- Can run? yes

## Commands
- /review

## Artifacts
- $ReviewReportPath

## Pass criteria
- Review file has SHIP/Reviewed/Branch lines

## Report path
- $ReviewReportPath

## Owner / role
- Reviewer
"@
}

Ensure-Dir $Inbox
Ensure-Dir $Outbox

$state = Join-Path $Outbox ("_" + $Role + "_done.txt")
if (-not (Test-Path $state)) { New-Item -ItemType File -Path $state | Out-Null }

if ($Role -ne "courier") {
  $profile = $profiles[$Role]
  if (-not $profile) { throw "Unknown role: $Role" }

  $codex = Get-Command codex.cmd -ErrorAction SilentlyContinue
  if (-not $codex) { $codex = Get-Command codex -ErrorAction SilentlyContinue }
  if (-not $codex) { throw "codex executable not found in PATH" }

  while ($true) {
    $tasks = Get-ChildItem $Inbox -Filter *.md | Sort-Object LastWriteTime
    foreach ($t in $tasks) {
      $done = Get-Content $state -ErrorAction SilentlyContinue
      if ($done -contains $t.Name) { continue }

      $text = Get-Content $t.FullName -Raw
      if (-not (Role-Matches $text $Role)) {
        if (-not (Get-OwnerRole $text)) {
          Write-Host "NO OWNER: $($t.Name)"
        }
        continue
      }

      $report = Get-ReportPath $text
      if (-not $report) {
        Write-Host "NO REPORT PATH: $($t.Name)"
        continue
      }

      Ensure-Dir (Split-Path $report)
      Write-Host "RUN: $($t.Name) -> $report"
      $output = & $codex.Source exec --profile $profile $text
      $output | Set-Content -Path $report -Encoding utf8
      Add-Content -Path $state -Value $t.Name

      if ($Role -eq "guard") {
        $auto = Get-AutoReview $text
        if ($auto.Enable) {
          $taskId = [System.IO.Path]::GetFileNameWithoutExtension($t.Name)
          $repoRoot = (Get-Location).Path
          $reviewTaskRel = $auto.TaskPath
          $reviewReportRel = $auto.ReportPath
          if (-not $reviewTaskRel) { $reviewTaskRel = ("tasks/inbox/" + $taskId + "_review.md") }
          if (-not $reviewReportRel) { $reviewReportRel = ("tasks/outbox/" + $taskId + "_review.md") }
          $reviewTaskPath = Resolve-PathSafe $reviewTaskRel $repoRoot
          $reviewReportPath = Resolve-PathSafe $reviewReportRel $repoRoot
          $scope = Get-ReviewScope $text
          if (-not (Test-Path $reviewTaskPath)) {
            Ensure-Dir (Split-Path $reviewTaskPath)
            $content = New-ReviewerTaskContent $taskId $t.FullName $report $scope.Base $scope.Head $reviewReportRel
            $content | Set-Content -Path $reviewTaskPath -Encoding utf8
            Write-Host "AUTO-REVIEW TASK: $reviewTaskPath"
          }
        }
      }
    }
    Start-Sleep -Seconds $PollSeconds
  }
} else {
  while ($true) {
    $reviews = Get-ChildItem $Outbox -Filter *_review.md | Sort-Object LastWriteTime
    foreach ($r in $reviews) {
      $done = Get-Content $state -ErrorAction SilentlyContinue
      if ($done -contains $r.Name) { continue }

      $text = Get-Content $r.FullName -Raw
      $ship = $text -match '(?im)^SHIP:\s*yes\b'
      $reviewed = $text -match '(?im)^Reviewed:\s*/review used \(yes\)'
      $branchMatch = [regex]::Match($text, '(?im)^Branch:\s*(\S+)')

      if (-not $ship -or -not $reviewed -or -not $branchMatch.Success) {
        Write-Host "NOT READY: $($r.Name)"
        continue
      }

      $branch = $branchMatch.Groups[1].Value
      Write-Host "AUTO PUSH/PR: $branch ($($r.Name))"

      & git push -u origin $branch
      if ($LASTEXITCODE -ne 0) {
        Write-Host "git push failed"
        continue
      }

      & gh pr view --head $branch --json url > $null 2>&1
      if ($LASTEXITCODE -ne 0) {
        $title = "Auto PR: $branch"
        $body = "Auto-created by courier after SHIP: yes / Reviewed: yes."
        & gh pr create --title $title --body $body
      } else {
        Write-Host "PR already exists for $branch"
      }

      Add-Content -Path $state -Value $r.Name
    }
    Start-Sleep -Seconds $PollSeconds
  }
}
