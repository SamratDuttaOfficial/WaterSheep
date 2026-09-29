@echo off
rem Creates .venv (and a private Python in .python if none is found).
setlocal EnableExtensions EnableDelayedExpansion
set "ROOT=%~dp0"
set "PYDIR=%ROOT%.python"
set "VENV=%ROOT%.venv"
set "VPY=%VENV%\Scripts\python.exe"
set "PYVER=3.12.7"
set "PBSTAG=20241016"

if exist "%VPY%" exit /b 0

set "PY="
call :try "%PYDIR%\python.exe"
if not defined PY call :try python
if not defined PY call :try py
if not defined PY goto :download
set "FOUND="
for /f "tokens=2" %%v in ('call "%PY%" --version 2^>^&1') do set "FOUND=%%v"
echo [setup] using the Python already on this machine: %PY% %FOUND%
goto :makevenv

:download
echo [setup] no Python 3.10+ found - downloading a private CPython into %PYDIR%
if "%PROCESSOR_ARCHITECTURE%"=="ARM64" (set "TRIPLE=aarch64-pc-windows-msvc") else (set "TRIPLE=x86_64-pc-windows-msvc")
set "WS_TRIPLE=%TRIPLE%"
set "WS_TGZ=%TEMP%\watersheep-cpython-%TRIPLE%.tar.gz"
set "WS_TAG=%PBSTAG%"
set "WS_VER=%PYVER%"
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ErrorActionPreference='Stop'; $t=$env:WS_TAG; $v=$env:WS_VER; $tr=$env:WS_TRIPLE; $u='https://github.com/astral-sh/python-build-standalone/releases/download/'+$t+'/cpython-'+$v+'+'+$t+'-'+$tr+'-install_only.tar.gz'; try { $h=@{'User-Agent'='watersheep-setup'}; $r=Invoke-RestMethod 'https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest' -Headers $h; $a=$r.assets | Where-Object { $_.name -like ('cpython-3.12.*-'+$tr+'-install_only.tar.gz') } | Select-Object -First 1; if ($a) { $u=$a.browser_download_url } } catch { }; Write-Host ('[setup] ' + $u); Invoke-WebRequest $u -OutFile $env:WS_TGZ -UseBasicParsing"
if errorlevel 1 goto :nopython
if not exist "%WS_TGZ%" goto :nopython
if exist "%PYDIR%_tmp" rmdir /s /q "%PYDIR%_tmp"
mkdir "%PYDIR%_tmp"
tar -xzf "%WS_TGZ%" -C "%PYDIR%_tmp"
if errorlevel 1 goto :nopython
if exist "%PYDIR%" rmdir /s /q "%PYDIR%"
move "%PYDIR%_tmp\python" "%PYDIR%" >nul
rmdir /s /q "%PYDIR%_tmp" 2>nul
del "%WS_TGZ%" 2>nul
set "PY=%PYDIR%\python.exe"
if not exist "%PY%" goto :nopython
echo [setup] private Python ready: %PY%

:makevenv
echo [setup] creating the virtual environment in %VENV%
"%PY%" -m venv --system-site-packages "%VENV%"
if errorlevel 1 (
    echo [setup] could not create the virtual environment.
    exit /b 1
)
"%VPY%" -m pip install -q --disable-pip-version-check --upgrade pip wheel
echo [setup] ready. run.py installs the heavier packages itself when first needed.
exit /b 0

:try
if defined PY goto :eof
"%~1" -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>&1
if errorlevel 1 goto :eof
set "PY=%~1"
goto :eof

:nopython
echo.
echo [setup] Could not download a private Python automatically.
echo [setup] Install Python 3.10+ from https://www.python.org/downloads/ and run this again.
exit /b 1
