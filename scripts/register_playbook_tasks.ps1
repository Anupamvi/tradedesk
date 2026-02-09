param(
    [string]$DailyTime = '16:45',
    [string]$WeeklyTime = '17:15',
    [string]$MonthlyTime = '18:00',
    [string]$TaskPrefix = 'UW_TradePlaybook'
)

$ErrorActionPreference = 'Stop'

$dailyTask = "${TaskPrefix}_Daily"
$weeklyTask = "${TaskPrefix}_Weekly"
$monthlyTask = "${TaskPrefix}_Monthly"

$dailyCmd = 'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\uw_root\scripts\run_playbook_daily.ps1"'
$weeklyCmd = 'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\uw_root\scripts\run_playbook_weekly.ps1"'
$monthlyCmd = 'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "C:\uw_root\scripts\run_playbook_monthly.ps1"'

Write-Host "Registering $dailyTask @ $DailyTime (daily)"
schtasks /Create /F /SC DAILY /TN $dailyTask /TR $dailyCmd /ST $DailyTime /RL LIMITED

Write-Host "Registering $weeklyTask @ $WeeklyTime (friday)"
schtasks /Create /F /SC WEEKLY /D FRI /TN $weeklyTask /TR $weeklyCmd /ST $WeeklyTime /RL LIMITED

Write-Host "Registering $monthlyTask @ $MonthlyTime (day 1)"
schtasks /Create /F /SC MONTHLY /D 1 /TN $monthlyTask /TR $monthlyCmd /ST $MonthlyTime /RL LIMITED

Write-Host "Done."
schtasks /Query /TN $dailyTask /FO LIST /V
schtasks /Query /TN $weeklyTask /FO LIST /V
schtasks /Query /TN $monthlyTask /FO LIST /V
