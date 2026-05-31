@echo off
echo ========================================================
echo   DeepScan Forensics - Daily Training Session Launcher
echo ========================================================
echo.
echo Starting training loop with RTX 4060...
echo This will automatically resume from the latest checkpoint if it exists.
echo.

set PYTHONPATH=%cd%
python ml\training\train.py --resume

echo.
echo Training session complete. Checkpoint saved.
pause
