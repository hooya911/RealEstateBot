@echo off
title Real Estate Bot
color 0A
cd /d "%~dp0"

echo.
echo ================================================
echo   REAL ESTATE WALKTHROUGH BOT — STARTING...
echo ================================================
echo.

echo [1/2] Checking dependencies...
pip install -r requirements.txt -q
echo       Done.
echo.

echo [2/2] Starting bot...
echo       The bot is now LIVE and waiting in Telegram.
echo       To STOP the bot, press Ctrl+C in this window.
echo.
echo ================================================
echo.

python bot.py

echo.
echo Bot has stopped. Press any key to close.
pause >nul
