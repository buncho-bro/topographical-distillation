@echo off
title Concept Forge WebUI Launcher
color 0A

echo ==================================================
echo       Concept Forge: Isolated Concept Distiller
echo ==================================================
echo.

:: 1. Check if Python is installed
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not installed or not added to PATH.
    echo Please install Python 3.10 or higher and try again.
    pause
    exit /b
)

:: 2. Check and create virtual environment
if not exist ".venv\Scripts\activate" (
    echo [INFO] Creating Python virtual environment ^(.venv^)...
    python -m venv .venv
    if %ERRORLEVEL% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b
    )
)

:: 3. Activate virtual environment
echo [INFO] Activating virtual environment...
call .venv\Scripts\activate

:: 4. Install dependencies
echo [INFO] Checking and installing dependencies from requirements.txt...
echo This might take a few minutes if this is your first run.
pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Failed to install dependencies. Please check your internet connection.
    pause
    exit /b
)

:: 5. Launch the Web UI
echo.
echo ==================================================
echo      Starting Concept Forge WebUI...
echo ==================================================
echo.
python concept_forge_webui.py

:: 6. Keep window open if app crashes
echo.
echo [INFO] Application has stopped.
pause
