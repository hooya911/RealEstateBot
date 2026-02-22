"""
config.py — Centralised configuration loaded from .env
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── API credentials ────────────────────────────────────────────────────────────
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
OPENAI_API_KEY: str     = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY: str     = os.getenv("GOOGLE_API_KEY", "")

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

# ── Whisper ────────────────────────────────────────────────────────────────────
# initial_prompt steers Whisper toward bilingual Farsi/English real-estate vocab.
# Language is intentionally NOT forced so mixed speech transcribes naturally.
WHISPER_INITIAL_PROMPT: str = (
    "این یک ویدیو از بازدید املاک است. "
    "The property features include: MLS number, Master Bedroom, Ensuite Bathroom, "
    "Open Concept Kitchen, Hardwood Floors, and Backyard. "
    "Please transcribe both the Farsi and English technical terms accurately."
)

# ── Gemini ─────────────────────────────────────────────────────────────────────
GEMINI_MODEL: str = "gemini-2.5-flash-preview-05-20"   # primary — auto-falls back to GPT-4o if unavailable

# ── File handling ──────────────────────────────────────────────────────────────
TEMP_DIR: str = "temp"

# Telegram Bot API download limit (free tier)
TELEGRAM_MAX_FILE_MB: int = 20
