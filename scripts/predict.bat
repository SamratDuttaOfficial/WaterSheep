@echo off
rem Asks the trained model. Arguments pass through to watersheep.cli.
call "%~dp0setup.bat"
if errorlevel 1 exit /b 1
setlocal
set "PYTHONPATH=%~dp0.."
"%~dp0..\.venv\Scripts\python.exe" -m watersheep.cli %*
