@echo off
setlocal
cd /d "%~dp0"
title ISL Counter (web)

if not exist ".venv_web\Scripts\python.exe" (
    echo Creating the Python environment ^(first run only^) ...
    where py >nul 2>nul && (py -3 -m venv .venv_web) || (python -m venv .venv_web)
    if not exist ".venv_web\Scripts\python.exe" (
        echo ERROR: Python 3 was not found. Install it from python.org ^(tick "Add python.exe to PATH"^).
        pause & exit /b 1
    )
)
".venv_web\Scripts\python.exe" -c "import flask, numpy" >nul 2>nul
if errorlevel 1 (
    echo Installing Flask and NumPy ...
    ".venv_web\Scripts\python.exe" -m pip install -r requirements.txt || (echo Install failed. & pause & exit /b 1)
)

rem open the browser a moment after the server starts
start "" cmd /c "timeout /t 2 >nul & start http://127.0.0.1:5000"
".venv_web\Scripts\python.exe" app.py
pause
