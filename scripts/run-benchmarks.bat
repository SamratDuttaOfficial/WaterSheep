@echo off
rem Benchmarks the newest export (or --model). Arguments pass through to watersheep.benchmark.
call "%~dp0setup.bat"
if errorlevel 1 exit /b 1
setlocal
set "PYTHONPATH=%~dp0.."
"%~dp0..\.venv\Scripts\python.exe" -m watersheep.benchmark %*
set "RC=%errorlevel%"
echo %cmdcmdline% | findstr /i /c:"%~nx0" >nul && pause
exit /b %RC%
