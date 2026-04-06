@echo off
REM Trade Monitor — Windows Task Scheduler setup
REM Creates a scheduled task that runs every 30 min during market hours (Mon-Fri 9:30-16:00 ET)

echo Setting up Trade Monitor scheduled task...

REM Delete existing task if present
schtasks /delete /tn "TradeMonitor" /f >nul 2>&1

REM Create task: runs every 30 minutes
schtasks /create ^
  /tn "TradeMonitor" ^
  /tr "python -m uwos.trade_monitor --force" ^
  /sc minute /mo 30 ^
  /st 09:30 ^
  /et 16:05 ^
  /sd %date% ^
  /d MON,TUE,WED,THU,FRI ^
  /rl HIGHEST ^
  /f

if %errorlevel% equ 0 (
    echo.
    echo Trade Monitor scheduled task created successfully!
    echo   Frequency: Every 30 minutes
    echo   Hours: 9:30 AM - 4:05 PM
    echo   Days: Monday - Friday
    echo   Working dir: c:\uw_root
    echo.
    echo To test: schtasks /run /tn "TradeMonitor"
    echo To remove: schtasks /delete /tn "TradeMonitor" /f
) else (
    echo.
    echo FAILED - try running this script as Administrator
)
pause
