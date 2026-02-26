@echo off
title NASA CMAPSS Engine Health Monitor
color 0B
echo.
echo  ============================================================
echo   NASA CMAPSS  --  Engine Health Monitor
echo   DRDO / GTRE  Turbofan PHM System
echo  ============================================================
echo.
echo  Starting web dashboard...
echo  It will open automatically in your browser.
echo  To stop: press Ctrl+C in this window
echo.

cd /d "%~dp0"

if not exist "venv\Scripts\activate.bat" (
    echo  [ERROR] Virtual environment not found.
    echo  Please run scripts\setup.bat first.
    pause
    exit /b 1
)

call venv\Scripts\activate.bat

pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo  Installing Streamlit...
    pip install streamlit plotly scikit-learn --quiet
)

streamlit run app.py --server.port 8501 --browser.gatherUsageStats false ^
    --theme.base dark ^
    --theme.primaryColor "#7eb8f7" ^
    --theme.backgroundColor "#0a0e1a" ^
    --theme.secondaryBackgroundColor "#0d1a2e" ^
    --theme.textColor "#c8d6e5"
