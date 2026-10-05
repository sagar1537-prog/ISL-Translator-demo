@echo off
setlocal
cd /d "%~dp0"
title ISL Translator Mk III - learning your signs
if not exist ".venv\Scripts\python.exe" (echo Run setup.bat first. & pause & exit /b 1)
if not exist "dataset_path.txt" (echo Run train.bat once first. & pause & exit /b 1)
set /p DATA=<"dataset_path.txt"
echo Teaching the model your recorded signs (a few minutes) ...
".venv\Scripts\python.exe" train.py --data "%DATA%" --finetune %*
if errorlevel 1 (echo. & echo Stopped with an error ^(see above^). & pause & exit /b 1)
echo.
echo Done. Reload the page in your browser (or start run.bat) to use the updated model.
pause
