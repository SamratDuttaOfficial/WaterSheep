@echo off
rem Runs or resumes the pipeline. Arguments pass through to watersheep.run.
call "%~dp0setup.bat"
if errorlevel 1 exit /b 1
setlocal
set "PYTHONPATH=%~dp0.."
"%~dp0..\.venv\Scripts\python.exe" -m watersheep.run %*
