@echo off
setlocal enabledelayedexpansion

REM Create desktop shortcut for Neural Ecosystem Simulator 3D
powershell -Command ^
"$DesktopPath = [Environment]::GetFolderPath('Desktop'); ^
$ShortcutPath = Join-Path $DesktopPath 'Neural Ecosystem Simulator 3D.lnk'; ^
$WshShell = New-Object -ComObject WScript.Shell; ^
$Shortcut = $WshShell.CreateShortcut($ShortcutPath); ^
$Shortcut.TargetPath = '%~dp0start.bat'; ^
$Shortcut.WorkingDirectory = '%~dp0'; ^
$Shortcut.Description = 'Neural Ecosystem Simulator 3D'; ^
$Shortcut.WindowStyle = 1; ^
$Shortcut.Save(); ^
Write-Host 'Shortcut created on Desktop!'"

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [SUCCESS] Desktop shortcut created!
    echo You can now launch the app from your desktop.
) else (
    echo [WARNING] Could not create desktop shortcut.
    echo You can still use: start.bat
)

pause
