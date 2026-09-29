@echo off
rem Runs or resumes the pipeline. Arguments pass through to run.py.
call "%~dp0setup.bat"
if errorlevel 1 exit /b 1
"%~dp0.venv\Scripts\python.exe" "%~dp0run.py" %*
