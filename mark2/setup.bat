@echo off
setlocal
cd /d "%~dp0"
title ISL Translator Mk II - setup
echo ============================================
echo   ISL Translator Mk II  -  one-time setup
echo ============================================
echo.

where py >nul 2>nul && (set "PYLAUNCH=py -3") || (set "PYLAUNCH=python")
%PYLAUNCH% -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python 3.9 or newer was not found.
    echo Install Python 3.11 or 3.12 from python.org and tick "Add python.exe to PATH", then run setup.bat again.
    pause & exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/4] Creating the Python environment ...
    %PYLAUNCH% -m venv .venv || (echo Could not create the environment. & pause & exit /b 1)
) else (
    echo [1/4] Python environment already exists.
)
set "PY=.venv\Scripts\python.exe"

echo [2/4] Updating pip ...
"%PY%" -m pip install --upgrade pip --quiet

echo [3/4] Installing packages (PyTorch CPU build, about 200 MB, only the first time) ...
"%PY%" -m pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet
if errorlevel 1 (
    echo PyTorch CPU download failed, trying the standard package ...
    "%PY%" -m pip install torch --quiet || (echo Could not install PyTorch. Check the internet connection. & pause & exit /b 1)
)
"%PY%" -m pip install -r requirements-train.txt --quiet || (echo Package install failed. & pause & exit /b 1)

echo [4/4] Downloading the body-tracking model (6 MB) ...
if not exist "static\models\pose_landmarker_lite.task" (
    curl -fL --retry 3 -o "static\models\pose_landmarker_lite.task" "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    if errorlevel 1 (
        del "static\models\pose_landmarker_lite.task" >nul 2>nul
        echo   Could not download it. The app still works with hands only; run setup.bat again later to add it.
    )
) else (
    echo   Already downloaded.
)

echo.
echo Setup complete.
if not exist "static\model\model.json" (
    echo Next: run train.bat once to train on the Kaggle dataset, then run.bat.
) else (
    echo Next: run.bat
)
echo.
pause
