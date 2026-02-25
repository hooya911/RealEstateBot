"""
config.py — Centralised configuration loaded from .env
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── API credentials ────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN: str       = os.getenv("TELEGRAM_BOT_TOKEN", "")
OPENAI_API_KEY: str           = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY: str           = os.getenv("GOOGLE_API_KEY", "")

# Google Cloud service-account JSON — used by Chirp 3 (Speech-to-Text v2)
# Same credential format as telegram-audio-bot's GOOGLE_CLOUD_CREDENTIALS
GOOGLE_CLOUD_CREDENTIALS: str = os.getenv("GOOGLE_CLOUD_CREDENTIALS", "")

# Brave Search API — used to look up the property address from the MLS number.
# Get a free key (5 000 req/month) at: https://brave.com/search/api/
BRAVE_API_KEY: str      = os.getenv("BRAVE_API_KEY", "")

# ── Security: whitelist of Telegram user IDs that may use the bot ──────────────
# If ALLOWED_USER_IDS is empty in .env, the bot is open to everyone.
_raw_ids = os.getenv("ALLOWED_USER_IDS", "")
ALLOWED_USER_IDS: list[int] = [
    int(uid.strip())
    for uid in _raw_ids.split(",")
    if uid.strip().isdigit()
]

# ── Gemini ─────────────────────────────────────────────────────────────────────
GEMINI_MODEL: str = "gemini-2.5-flash-preview-05-20"   # primary — auto-falls back to GPT-4o if unavailable

# ── File handling ──────────────────────────────────────────────────────────────
TEMP_DIR: str = "temp"

# Telegram Bot API download limit (free tier)
TELEGRAM_MAX_FILE_MB: int = 20
