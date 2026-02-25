@echo off
REM ============================================================
REM  NASA CMAPSS — Windows Setup Script
REM  Double-click this file OR run in CMD / PowerShell:
REM    scripts\setup.bat
REM ============================================================

echo.
echo ==================================================
echo   NASA CMAPSS - VS Code Setup (Windows)
echo ==================================================
echo.

REM Check we are in project root
if not exist "main.py" (
    echo ERROR: Run this from the project root folder
    pause
    exit /b 1
)

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install from https://python.org
    pause
    exit /b 1
)
echo [OK] Python found
python --version

REM Create venv
echo.
echo Creating virtual environment...
if exist "venv\" (
    echo [SKIP] venv already exists
) else (
    python -m venv venv
    echo [OK] venv created
)

REM Activate
echo.
echo Activating venv...
call venv\Scripts\activate.bat
echo [OK] venv activated

REM Upgrade pip
echo.
echo Upgrading pip...
pip install --upgrade pip --quiet
echo [OK] pip upgraded

REM Install dependencies
echo.
echo Installing dependencies (2-5 min)...
pip install -r requirements.txt
echo [OK] All packages installed

REM Jupyter kernel
echo.
echo Registering Jupyter kernel...
python -m ipykernel install --user --name cmapss-env --display-name "Python (cmapss-env)"
echo [OK] Kernel registered

REM Directories
mkdir reports\logs 2>nul
mkdir reports\figures 2>nul
echo [OK] Output directories ready

REM Git init
echo.
if exist ".git\" (
    echo [SKIP] Git already initialised
) else (
    git init
    git add .
    git commit -m "Initial commit - NASA CMAPSS sensor fault detection"
    echo [OK] Git repository created
)

REM Done
echo.
echo ==================================================
echo   SETUP COMPLETE!
echo ==================================================
echo.
echo   ACTIVATE (every new terminal):
echo     venv\Scripts\activate
echo.
echo   TRAIN THE MODEL:
echo     python main.py train
echo.
echo   FULL PIPELINE:
echo     python main.py all
echo.
echo   EVALUATE ONLY:
echo     python main.py evaluate
echo.
echo   OPEN NOTEBOOK:
echo     jupyter notebook notebooks\NASA_CMAPSS_Complete.ipynb
echo.
echo   RUN TESTS:
echo     pytest tests\ -v
echo.
echo   IN VS CODE:
echo     1. Open folder sensor-ml-project-v2
echo     2. Press F5, select Run Full Pipeline
echo.
pause
