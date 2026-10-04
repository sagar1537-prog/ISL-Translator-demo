@echo off
setlocal
cd /d "%~dp0"
title ISL Translator Mk II
if not exist ".venv\Scripts\python.exe" (echo Run setup.bat first. & pause & exit /b 1)
if not exist "static\model\model.json" (
    echo No trained model yet. Run train.bat first ^(it uses your Kaggle dataset^).
    pause & exit /b 1
)
if "%PORT%"=="" set "PORT=8000"
rem open the browser a moment after the server starts
start "" cmd /c "timeout /t 2 >nul & start http://127.0.0.1:%PORT%"
".venv\Scripts\python.exe" app.py
pause
