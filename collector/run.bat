@echo off
cd /d "%~dp0"
set QT_QPA_PLATFORM_PLUGIN_PATH=%~dp0venv\Lib\site-packages\PyQt5\Qt5\plugins\platforms
venv\Scripts\python.exe main.py
pause
