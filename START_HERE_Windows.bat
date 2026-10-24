@echo off
rem ==========================================================================
rem  Person Search - one-click setup and start for Windows 10/11.
rem  Double-click this file. The first run installs everything it needs
rem  (about 10-20 minutes, needs internet). Later runs start in seconds.
rem ==========================================================================
setlocal EnableExtensions
title Person Search
cd /d "%~dp0"
set "PYTHONUTF8=1"
set "PYTHONNOUSERSITE=1"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"

if not exist "requirements.txt" goto :not_extracted

rem The Python environment lives in a short path outside the project folder,
rem so long Windows paths and spaces in the folder name never cause trouble.
set "APPDIR=%LOCALAPPDATA%\PersonSearch"
set "VENV=%APPDIR%\venv"
set "VPY=%VENV%\Scripts\python.exe"
set "STAMP=%VENV%\installed-requirements.txt"

if not exist "%VPY%" goto :setup
if not exist "%STAMP%" goto :setup
fc /b "%STAMP%" "requirements.txt" >nul 2>&1
if errorlevel 1 goto :setup
goto :run


:setup
echo.
echo  ==============================================================
echo   First-time setup. This takes 10-20 minutes and needs internet.
echo   Please keep this window open until the app opens.
echo  ==============================================================
echo.
if not exist "%APPDIR%" mkdir "%APPDIR%"

echo [1/4] Checking for Python 3.11 ...
call :find_python
if defined PY goto :have_python
echo       Python 3.11 not found. Downloading it (about 25 MB) ...
set "PYINST=%TEMP%\python-3.11.9-amd64.exe"
call :download "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe" "%PYINST%"
if not exist "%PYINST%" goto :fail_download
echo       Installing Python 3.11 for this user only (no admin needed) ...
"%PYINST%" /quiet InstallAllUsers=0 PrependPath=0 Include_launcher=0 Include_test=0 Include_doc=0 Shortcuts=0 AssociateFiles=0
call :find_python
if not defined PY goto :fail_python
:have_python
echo       Using %PY%

echo [2/4] Checking the Microsoft Visual C++ runtime (needed by PyTorch) ...
call :check_vcredist
if not errorlevel 1 goto :have_vcredist
echo       Not found. Downloading and installing it.
echo       If Windows asks "Do you want to allow this app to make changes", click YES.
set "VCINST=%TEMP%\vc_redist.x64.exe"
call :download "https://aka.ms/vs/17/release/vc_redist.x64.exe" "%VCINST%"
if exist "%VCINST%" "%VCINST%" /install /quiet /norestart
:have_vcredist

echo [3/4] Creating the Python environment ...
if not exist "%VPY%" "%PY%" -m venv "%VENV%"
if not exist "%VPY%" goto :fail_venv

echo [4/4] Installing packages (about 1 GB, the longest step) ...
"%VPY%" -m pip install --upgrade pip --timeout 120 --retries 5
"%VPY%" -m pip install -r requirements.txt --timeout 120 --retries 5
if not errorlevel 1 goto :pip_ok
echo       Some downloads failed. Trying once more ...
"%VPY%" -m pip install -r requirements.txt --timeout 120 --retries 5
if errorlevel 1 goto :fail_pip
:pip_ok

echo       Checking that everything loads ...
"%VPY%" -c "import torch, cv2, onnxruntime, open_clip, ultralytics, streamlit, motmetrics"
if errorlevel 1 goto :fail_import
copy /y "requirements.txt" "%STAMP%" >nul
echo.
echo  Setup complete.
echo.


:run
"%VPY%" launch.py
if errorlevel 1 goto :fail_run
goto :eof


rem ---------------------------------------------------------------- helpers
:find_python
set "PY="
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if defined PY goto :eof
if exist "%ProgramFiles%\Python311\python.exe" set "PY=%ProgramFiles%\Python311\python.exe"
if defined PY goto :eof
for /f "delims=" %%i in ('py -3.11 -c "import sys; print(sys.executable)" 2^>nul') do set "PY=%%i"
goto :eof

:check_vcredist
reg query "HKLM\SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" /v Installed >nul 2>&1 && exit /b 0
reg query "HKLM\SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64" /v Installed >nul 2>&1 && exit /b 0
exit /b 1

:download
curl -L --fail --silent --show-error -o "%~2" "%~1"
if exist "%~2" goto :eof
powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol='Tls12'; Invoke-WebRequest -UseBasicParsing -Uri '%~1' -OutFile '%~2'"
goto :eof


rem ---------------------------------------------------------------- errors
:not_extracted
echo.
echo  This file was opened from inside the ZIP.
echo  Right-click the ZIP file, choose "Extract All...", open the extracted
echo  folder and double-click START_HERE_Windows.bat again.
goto :stop

:fail_download
echo.
echo  Could not download Python. Check the internet connection and try again.
echo  (Or install Python 3.11 from https://www.python.org/downloads/ and re-run.)
goto :stop

:fail_python
echo.
echo  Python 3.11 could not be installed automatically.
echo  Install it from https://www.python.org/downloads/release/python-3119/
echo  ("Windows installer (64-bit)"), then double-click this file again.
goto :stop

:fail_venv
echo.
echo  Could not create the Python environment in %VENV%
goto :stop

:fail_pip
echo.
echo  Package installation failed - usually a dropped internet connection.
echo  Double-click this file again; packages already downloaded are kept.
goto :stop

:fail_import
echo.
echo  The packages installed but do not load. Restart the computer (this finishes
echo  the Visual C++ runtime install) and double-click this file again.
goto :stop

:fail_run
echo.
echo  The app stopped with an error (see the messages above).

:stop
echo.
pause
exit /b 1
