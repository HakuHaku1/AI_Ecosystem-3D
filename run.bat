@echo off
title Neural Ecosystem Simulator 3D
cd /d "%~dp0"

echo Launching Ecosystem...

:: Try 'py' launcher first (standard for Windows), then fallback to 'python'
where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    py neural_system_3d.py
) else (
    where python >nul 2>nul
    if %ERRORLEVEL% EQU 0 (
        python neural_system_3d.py
    ) else (
        echo [ERROR] Python was not found on your system.
        echo Please install Python from python.org and ensure 'Add Python to PATH' is checked.
    )
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] The simulation exited with code %ERRORLEVEL%.
    pause
)