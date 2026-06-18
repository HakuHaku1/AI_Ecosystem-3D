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

:: Check Python version is 3.11 (required for compatibility)
for /f "tokens=2" %%i in ('!PYTHON_CMD! --version 2^>^&1') do set VERSION_NUM=%%i
for /f "tokens=1,2 delims=." %%a in ("!VERSION_NUM!") do (
    set MAJOR=%%a
    set MINOR=%%b
)
if !MAJOR! LSS 3 (
    echo [ERROR] Python 3.11 is required. Found: !PYTHON_VERSION!
    pause
    exit /b 1
)
if !MAJOR! EQU 3 if !MINOR! LSS 11 (
    echo [ERROR] Python 3.11 is required. Found: !PYTHON_VERSION!
    pause
    exit /b 1
)

:: Install/upgrade pip
echo.
echo [2/3] Installing required dependencies...
!PYTHON_CMD! -m pip install --upgrade "pip>=24.0" "setuptools>=70,<82" "wheel>=0.40" --quiet
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Failed to upgrade pip, attempting to continue...
)

:: Install required packages
set PACKAGES=numpy cython torch moderngl pygame
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

exit /b %ERRORLEVEL%

:: ============================================
:: Function: Download and Install Python
:: ============================================
:download_and_install_python
echo.
echo Downloading Python 3.12 installer...

:: Detect system architecture
if "%PROCESSOR_ARCHITECTURE%"=="AMD64" (
    set PYTHON_INSTALLER=python-3.12.0-amd64.exe
    set DOWNLOAD_URL=https://www.python.org/ftp/python/3.12.0/python-3.12.0-amd64.exe
) else (
    set PYTHON_INSTALLER=python-3.12.0.exe
    set DOWNLOAD_URL=https://www.python.org/ftp/python/3.12.0/python-3.12.0.exe
)

set INSTALLER_PATH=%TEMP%\%PYTHON_INSTALLER%

echo Downloading from: %DOWNLOAD_URL%
powershell -Command "(New-Object Net.WebClient).DownloadFile('%DOWNLOAD_URL%', '%INSTALLER_PATH%')" 2>nul
if not exist "%INSTALLER_PATH%" (
    echo [ERROR] Failed to download Python installer.
    echo Please download manually from: https://www.python.org/downloads/
    exit /b 1
)

echo Running Python installer...
echo Please wait - this may take a few minutes...
"%INSTALLER_PATH%" /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
if %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Installer returned code %ERRORLEVEL%, but may still have succeeded.
)

:: Clean up installer
del "%INSTALLER_PATH%" >nul 2>&1

:: Re-detect Python
timeout /t 2 /nobreak >nul
where py >nul 2>nul
if %ERRORLEVEL% EQU 0 (
    set PYTHON_CMD=py
    echo Python installed successfully!
    exit /b 0
) else (
    where python >nul 2>nul
    if %ERRORLEVEL% EQU 0 (
        set PYTHON_CMD=python
        echo Python installed successfully!
        exit /b 0
    )
)

echo [ERROR] Python installation verification failed.
echo Please restart your terminal and try again.
exit /b 1
