param(
    [string]$RepoRoot = 'C:\uw_root',
    [string]$SheetCsvUrl = 'https://docs.google.com/spreadsheets/d/1CXCsBHXVh99PNd7CXjsoqTK74nQEG7BJ/export?format=csv&gid=1135109247',
    [ValidateSet('all','yellow')]
    [string]$SheetRowFilter = 'yellow',
    [string]$OutDir = 'C:\uw_root\out\playbook_auto',
    [string]$ConfigPath = 'C:\uw_root\uwos\rulebook_config.yaml'
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$logDir = Join-Path $OutDir 'logs'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$stamp = Get-Date -Format 'yyyy-MM-dd_HHmmss'
$logPath = Join-Path $logDir "daily_$stamp.log"

$cmd = "python -m uwos.daily_risk_monitor --sheet-csv-url `"$SheetCsvUrl`" --sheet-row-filter `"$SheetRowFilter`" --config `"$ConfigPath`" --out-dir `"$OutDir`""

"[$(Get-Date -Format s)] Running: $cmd" | Out-File -FilePath $logPath -Encoding utf8
Invoke-Expression $cmd *>> $logPath
if ($LASTEXITCODE -ne 0) {
    "[$(Get-Date -Format s)] FAILED exit_code=$LASTEXITCODE" | Out-File -FilePath $logPath -Append -Encoding utf8
    exit $LASTEXITCODE
}
"[$(Get-Date -Format s)] DONE" | Out-File -FilePath $logPath -Append -Encoding utf8

