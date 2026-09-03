@echo off
rem One-click deploy/start entry (Windows). Double-click = full deploy.
rem Use "deploy.bat --check-only" for a check-only run.
rem Shared cross-platform logic lives in deploy.py.
rem Keep this file ASCII-only: cmd.exe parses .bat in the OEM codepage.
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
set "EXIT_CODE=1"

where py >nul 2>nul
if not errorlevel 1 (
  py -3.12 -c "pass" >nul 2>nul
  if not errorlevel 1 (
    py -3.12 deploy.py %*
    set "EXIT_CODE=!ERRORLEVEL!"
    goto :finish
  )
  py -3 -c "pass" >nul 2>nul
  if not errorlevel 1 (
    py -3 deploy.py %*
    set "EXIT_CODE=!ERRORLEVEL!"
    goto :finish
  )
)
where python >nul 2>nul
if not errorlevel 1 (
  python deploy.py %*
  set "EXIT_CODE=!ERRORLEVEL!"
  goto :finish
)
python3 deploy.py %*
set "EXIT_CODE=!ERRORLEVEL!"

:finish
echo.
echo ==============================
echo   Deploy finished. Press any key to close.
echo ==============================
pause >nul
endlocal & exit /b %EXIT_CODE%
