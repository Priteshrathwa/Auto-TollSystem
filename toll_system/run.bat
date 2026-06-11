@echo off
echo Starting AutoToll System...
echo.

REM Check if virtual environment is activated
python -c "import sys; print('Virtual environment:', sys.prefix != sys.base_prefix)"

REM Check if required packages are installed
echo Checking dependencies...
python -c "import flask, cv2, ultralytics, easyocr, mysql.connector" 2>nul
if errorlevel 1 (
    echo Error: Some required packages are not installed.
    echo Please run: pip install -r requirements.txt
    pause
    exit /b 1
)

echo Dependencies OK!
echo.

REM Start the application
echo Starting web server at http://localhost:5000
echo Press Ctrl+C to stop the server
echo.

python app/main.py

pause
