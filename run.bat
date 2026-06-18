@echo off
setlocal enabledelayedexpansion
title Neural Ecosystem Simulator 3D
cd /d "%~dp0"

echo ============================================
echo Neural Ecosystem Simulator 3D - Setup
echo ============================================
echo.

:: Check for Python installation
echo [1/3] Checking Python installation...
where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    for /f "tokens=*" %%i in ('py --version 2^>^&1') do set PYTHON_VERSION=%%i
    set PYTHON_CMD=py
    echo Found: !PYTHON_VERSION!
) else (
    where python >nul 2>nul
    if %ERRORLEVEL% EQU 0 (
        for /f "tokens=*" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
        set PYTHON_CMD=python
        echo Found: !PYTHON_VERSION!
    ) else (
        echo [ERROR] Python was not found on your system.
        echo Please install Python 3.8+ from https://www.python.org
        echo Make sure to check 'Add Python to PATH' during installation.
        pause
        exit /b 1
    )
)

for /f "tokens=2" %%i in ('!PYTHON_CMD! --version 2^>^&1') do set VERSION_NUM=%%i
for /f "tokens=1,2 delims=." %%a in ("!VERSION_NUM!") do (
    set MAJOR=%%a
    set MINOR=%%b
)
if !MAJOR! LSS 3 (
    echo [ERROR] Python 3.8 or higher is required. Found: !PYTHON_VERSION!
    pause
    exit /b 1
)
if !MAJOR! EQU 3 if !MINOR! LSS 8 (
    echo [ERROR] Python 3.8 or higher is required. Found: !PYTHON_VERSION!
    pause
    exit /b 1
)

echo.
echo [2/3] Installing required dependencies...
!PYTHON_CMD! -m pip install --upgrade pip setuptools wheel 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Failed to upgrade pip, attempting to continue...
)

:: Install required packages
set PACKAGES=numpy torch pygame moderngl cython
echo Installing: %PACKAGES%

for %%p in (%PACKAGES%) do (
    !PYTHON_CMD! -m pip install %%p --quiet
    if !ERRORLEVEL! NEQ 0 (
        echo [ERROR] Failed to install %%p
        echo Please ensure you have internet connection and sufficient disk space.
        pause
        exit /b 1
    )
)

echo Dependencies installed successfully!
echo.
echo [3/3] Launching Ecosystem...
echo ============================================
echo.

:: Run the application
!PYTHON_CMD! neural_system_3d.py
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] The simulation exited with code %ERRORLEVEL%.
    pause
)