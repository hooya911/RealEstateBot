@echo off
title Push to GitHub → Railway Auto-Deploy
cd /d "%~dp0"

echo.
echo ========================================
echo   Pushing to GitHub...
echo   Railway will auto-redeploy after this.
echo ========================================
echo.

git add .

:: Auto-generate commit message with timestamp
set TIMESTAMP=%DATE% %TIME%
git commit -m "Update: %TIMESTAMP%"

git push

echo.
echo ========================================
echo   Done! Railway is redeploying now.
echo   Check railway.app for the live logs.
echo ========================================
echo.
pause
