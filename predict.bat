@echo off
rem Asks the trained model. Arguments pass through to predict.py.
call "%~dp0setup.bat"
if errorlevel 1 exit /b 1
"%~dp0.venv\Scripts\python.exe" "%~dp0predict.py" %*
