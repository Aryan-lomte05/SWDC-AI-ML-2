@echo off
echo ========================================================
echo   DeepScan Forensics - Dataset & Training Pipeline
echo ========================================================
echo.
echo Phase 1: Downloading and preparing DFDC dataset (Capped at 10GB)...
echo.

set PYTHONPATH=%cd%
python "datasets\prepare_dfdc.py"

if %errorlevel% neq 0 (
    echo.
    echo ❌ Dataset preparation failed. Please check the logs above.
    pause
    exit /b %errorlevel%
)

echo.
echo Phase 2: Starting RTX 4060 Training Loop...
echo (This will auto-resume from the last saved epoch)
echo.

python "ml\training\train.py" --resume

echo.
echo Training session ended.
pause
