param(
    [string]$RepoRoot = 'C:\uw_root',
    [string]$SheetCsvUrl = 'https://docs.google.com/spreadsheets/d/1CXCsBHXVh99PNd7CXjsoqTK74nQEG7BJ/export?format=csv&gid=1135109247',
    [string]$OutDir = 'C:\uw_root\out\playbook_auto',
    [string]$ConfigPath = 'C:\uw_root\rulebook_config.yaml'
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$logDir = Join-Path $OutDir 'logs'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$stamp = Get-Date -Format 'yyyy-MM-dd_HHmmss'
$logPath = Join-Path $logDir "daily_$stamp.log"

$scriptPath = Join-Path $RepoRoot 'run_trade_playbook.py'
$cmd = "python `"$scriptPath`" daily --sheet-csv-url `"$SheetCsvUrl`" --config `"$ConfigPath`" --out-dir `"$OutDir`""

"[$(Get-Date -Format s)] Running: $cmd" | Out-File -FilePath $logPath -Encoding utf8
Invoke-Expression $cmd *>> $logPath
if ($LASTEXITCODE -ne 0) {
    "[$(Get-Date -Format s)] FAILED exit_code=$LASTEXITCODE" | Out-File -FilePath $logPath -Append -Encoding utf8
    exit $LASTEXITCODE
}
"[$(Get-Date -Format s)] DONE" | Out-File -FilePath $logPath -Append -Encoding utf8
