@echo off
rem Benchmarks the newest export (or --model). Arguments pass through to benchmark.py.
call "%~dp0setup.bat"
if errorlevel 1 exit /b 1
"%~dp0.venv\Scripts\python.exe" "%~dp0benchmark.py" %*
set "RC=%errorlevel%"
echo %cmdcmdline% | findstr /i /c:"%~nx0" >nul && pause
exit /b %RC%
