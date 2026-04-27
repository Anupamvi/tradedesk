@echo off
REM Trade Monitor — Windows Task Scheduler decommission
REM The active monitor now runs on the GCP VM via systemd timer.
REM Running this script deletes the old Windows task so ntfy alerts do not duplicate.

echo Decommissioning old Windows TradeMonitor scheduled task...

schtasks /query /tn "TradeMonitor" >nul 2>&1
if %errorlevel% equ 0 (
    schtasks /delete /tn "TradeMonitor" /f
    echo.
    echo Deleted old Windows TradeMonitor task.
) else (
    echo.
    echo No Windows TradeMonitor task was found.
)
echo.
echo Active monitor should be the GCP systemd timer only:
echo   trade-monitor.timer on tradedesk-monitor
pause
