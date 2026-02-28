param(
  [string]$Root = "C:\uw_root\out\replay_compare",
  [string]$Variant = "holistic_newstrategy_pf115_rebalanced_v3",
  [int]$LastN = 5
)

$ErrorActionPreference = "Stop"

if (-not (Get-Command code -ErrorAction SilentlyContinue)) {
  throw "VS Code CLI 'code' was not found on PATH."
}

if (-not (Test-Path $Root)) {
  throw "Root path not found: $Root"
}

$rows = @()
$variantDirs = Get-ChildItem -Path $Root -Recurse -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -eq $Variant }
foreach ($dir in $variantDirs) {
  $mds = Get-ChildItem -Path $dir.FullName -File -Filter "anu-expert-trade-table-*.md" -ErrorAction SilentlyContinue
  foreach ($m in $mds) {
    if ($m.Name -match "(\d{4}-\d{2}-\d{2})") {
      try { $d = [datetime]::ParseExact($matches[1], "yyyy-MM-dd", $null) } catch { continue }
      $rows += [pscustomobject]@{
        Date = $d
        File = $m.FullName
      }
    }
  }
}

if (-not $rows -or $rows.Count -eq 0) {
  throw "No recommendation markdown files found for variant '$Variant' under '$Root'."
}

$targets = $rows |
  Sort-Object -Property Date -Descending |
  Select-Object -First ([math]::Max(1, $LastN))

Write-Host "Opening $($targets.Count) recommendation file(s) in VS Code..." -ForegroundColor Cyan
foreach ($t in $targets) {
  Write-Host ("  " + $t.File)
  code $t.File
}
