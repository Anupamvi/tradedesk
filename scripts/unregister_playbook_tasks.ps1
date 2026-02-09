param(
    [string]$TaskPrefix = 'UW_TradePlaybook'
)

$ErrorActionPreference = 'Continue'
$names = @("${TaskPrefix}_Daily", "${TaskPrefix}_Weekly", "${TaskPrefix}_Monthly")
foreach ($n in $names) {
    schtasks /Delete /TN $n /F | Out-Null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Deleted $n"
    } else {
        Write-Host "Not found or failed: $n"
    }
}
