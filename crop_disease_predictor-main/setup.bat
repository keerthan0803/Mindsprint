@echo off
REM Crop Disease Predictor - Quick Setup Script for Windows

echo.
echo ======================================
echo Crop Disease Predictor - Setup Script
echo ======================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo Installing dependencies...
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo Error: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ======================================
echo Setup completed successfully!
echo ======================================
echo.
echo Next steps:
echo.
echo 1. Train the model (first time only):
echo    python train_model.py
echo.
echo 2. Run the web application:
echo    python app.py
echo.
echo 3. Open browser and navigate to:
echo    http://localhost:5000
echo.
pause
