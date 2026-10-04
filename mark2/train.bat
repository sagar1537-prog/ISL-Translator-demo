@echo off
setlocal
cd /d "%~dp0"
title ISL Translator Mk II - training
if not exist ".venv\Scripts\python.exe" (echo Run setup.bat first. & pause & exit /b 1)

set "DATA=C:\Users\VSJ\Downloads\archive\keypoints"
if exist "dataset_path.txt" set /p DATA=<"dataset_path.txt"
if not exist "%DATA%\" (
    echo Kaggle dataset folder not found: %DATA%
    set /p DATA=Paste the full path of the "keypoints" folder: 
)
set "DATA=%DATA:"=%"
if not exist "%DATA%\" (echo Folder not found: %DATA% & pause & exit /b 1)
>"dataset_path.txt" echo %DATA%

echo Training on: %DATA%
echo The first run reads all 4,000+ videos once (about 5-15 minutes), then trains (about 15-40 minutes on a laptop CPU).
echo Leave this window open. Progress is shown below.
echo.
".venv\Scripts\python.exe" train.py --data "%DATA%" %*
if errorlevel 1 (echo. & echo Training stopped with an error ^(see above^). & pause & exit /b 1)
echo.
echo Done. Start the app with run.bat
pause
