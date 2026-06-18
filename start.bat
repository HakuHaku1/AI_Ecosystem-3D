@echo off
setlocal enabledelayedexpansion
title Neural Ecosystem Simulator 3D
cd /d "%~dp0"

:: Try Python 3.11 first, then fall back to default python
py -3.11 --version >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    set PYTHON_CMD=py -3.11
    echo Using Python 3.11
) else (
    python --version >nul 2>nul
    if %ERRORLEVEL% EQU 0 (
        set PYTHON_CMD=python
        echo Using default Python
    ) else (
        echo [ERROR] Python not found. Please install Python 3.11
        pause
        exit /b 1
    )
)

echo Starting Neural Ecosystem Simulator 3D...
echo.

!PYTHON_CMD! neural_system_3d.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Application exited with code %ERRORLEVEL%
    pause
)

exit /b %ERRORLEVEL%
