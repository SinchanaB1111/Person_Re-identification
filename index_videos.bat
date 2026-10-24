@echo off
rem Indexes every video in data\videos (same as the "Index videos" button in the app).
rem Run START_HERE_Windows.bat once first so the Python environment exists.
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "VPY=%LOCALAPPDATA%\PersonSearch\venv\Scripts\python.exe"
if not exist "%VPY%" goto :no_setup
"%VPY%" index_videos.py %*
pause
goto :eof

:no_setup
echo Run START_HERE_Windows.bat first - it installs everything this needs.
pause
