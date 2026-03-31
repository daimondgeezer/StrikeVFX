@echo off
echo ============================================
echo  Atlas EP - Audio Visualizer Setup
echo ============================================

where python >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.9+ from python.org
    pause
    exit /b 1
)

echo Installing dependencies...
pip install -r requirements.txt

if errorlevel 1 (
    echo ERROR: pip install failed.
    pause
    exit /b 1
)

echo.
echo Starting Atlas EP...
python main.py
pause
