@echo off
setlocal enabledelayedexpansion

echo ===== OATFLAKE Setup and Startup Script =====
echo.

REM Check if Python is installed
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: Python is not installed or not in PATH.
    echo Please install Python 3.10+ and try again.
    goto :error
)

REM Create venv if it doesn't exist
if not exist .venv (
    echo Creating virtual environment...
    python -m venv .venv
    if %ERRORLEVEL% neq 0 (
        echo ERROR: Failed to create virtual environment.
        goto :error
    )
    echo Virtual environment created successfully.
) else (
    echo Virtual environment already exists.
)

REM Activate the virtual environment
echo Activating virtual environment...
call .venv\Scripts\activate.bat
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to activate virtual environment.
    goto :error
)

REM Install Poetry if not already installed
where poetry >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Installing Poetry...
    pip install poetry
    if %ERRORLEVEL% neq 0 (
        echo ERROR: Failed to install Poetry.
        goto :error
    )
)

REM Install dependencies using Poetry
echo Installing dependencies with Poetry...
poetry install
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to install dependencies.
    goto :error
)

REM Ensure essential packages are installed
echo Verifying essential packages...
pip install --quiet Jinja2 slack-bolt
if %ERRORLEVEL% neq 0 (
    echo ERROR: Failed to install essential packages.
    goto :error
)

REM Run the application
echo Starting the application...
python run.py
if %ERRORLEVEL% neq 0 (
    echo ERROR: Application exited with error code !ERRORLEVEL!
    goto :error
)

echo.
echo ===== Application Closed =====
goto :end

:error
echo.
echo ===== Error Occurred =====
echo.

:end
echo Press any key to exit...
pause >nul
