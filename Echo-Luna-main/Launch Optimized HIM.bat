@echo off
title H.I.M. Model - Optimized GUI Launcher
color 0A

echo.
echo ============================================================
echo    H.I.M. MODEL - OPTIMIZED GUI LAUNCHER
echo ============================================================
echo.
echo Starting optimized H.I.M. GUI with crash prevention...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python and try again.
    pause
    exit /b 1
)

REM Launch the optimized GUI
python launch_optimized_him.py

REM Keep window open if there was an error
if errorlevel 1 (
    echo.
    echo An error occurred. Check the messages above.
    pause
)

