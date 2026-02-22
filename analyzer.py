"""
analyzer.py — Single-purpose property info extractor

PRIMARY:  GPT-4o — best at identifying addresses and MLS numbers in
          bilingual Farsi/English real-estate audio transcriptions.
FALLBACK: Gemini (gemini-2.5-flash-preview) — auto-triggers on any GPT-4o error.

One public function:
    extract_address_and_mls(transcript) → (address | None, mls | None)

The transcript is the first 2 minutes of the walkthrough, where the agent
typically announces the property address and MLS number in English.
"""
import re
import logging

from google import genai
from google.genai import types
from openai import OpenAI

from config import GOOGLE_API_KEY, OPENAI_API_KEY, GEMINI_MODEL

logger = logging.getLogger(__name__)

_openai = OpenAI(api_key=OPENAI_API_KEY)          # PRIMARY  — GPT-4o
_gemini = genai.Client(api_key=GOOGLE_API_KEY)    # FALLBACK — Gemini


# ── Prompt ─────────────────────────────────────────────────────────────────────

_SYSTEM = (
    "You are a real estate data extraction assistant. "
    "Your only job is to find the property address and MLS number from a transcript. "
    "Be precise with numbers — never guess or invent digits."
)

_PROMPT = """You are reading the first 2 minutes of a real estate walkthrough recording.
At the start, the agent typically announces the property in English.

Extract exactly two things:

1. PROPERTY ADDRESS — The full street address.
   Examples: "123 Main Street", "188 Fairview Mall Drive Unit 506", "456 Oak Ave"
   - Include unit/suite number if mentioned.
   - Do NOT include city, province, or postal code.

2. MLS NUMBER — The listing number (7–9 digits).
   - It may be spoken as letters then digits: "MLS one two seven five four six one eight" → 12754618
   - It may be spoken digit by digit: "one two three four five six seven" → 1234567
   - Strip all spaces, dashes, and letters — return digits only.

TRANSCRIPT (first 2 minutes):
---
{transcript}
---

Reply in EXACTLY this format — two lines, nothing else:
ADDRESS: [full street address, or NOT_FOUND]
MLS: [digits only, or NOT_FOUND]"""


# ── Extraction ─────────────────────────────────────────────────────────────────

def extract_address_and_mls(transcript: str) -> tuple[str | None, str | None]:
    """
    Extract property address and MLS number from the opening of a walkthrough transcript.
    GPT-4o primary, Gemini fallback.

    Returns:
        (address, mls) — either value may be None if not found.
    """
    if not transcript or not transcript.strip():
        return None, None

    prompt = _PROMPT.format(transcript=transcript)

    # ── Primary: GPT-4o ────────────────────────────────────────────────────────
    try:
        r = _openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.0,
            max_tokens=80,
        )
        result = _parse_response(r.choices[0].message.content)
        logger.info("GPT-4o extraction → address=%r  mls=%r", result[0], result[1])
        return result

    except Exception as exc:
        logger.warning("GPT-4o extraction failed: %s — trying Gemini", exc)

    # ── Fallback: Gemini ───────────────────────────────────────────────────────
    try:
        r = _gemini.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM,
                temperature=0.0,
                max_output_tokens=80,
            ),
        )
        result = _parse_response(r.text)
        logger.info("Gemini extraction (fallback) → address=%r  mls=%r", result[0], result[1])
        return result

    except Exception as exc:
        logger.error("Gemini extraction also failed: %s", exc)
        return None, None


def _parse_response(raw: str) -> tuple[str | None, str | None]:
    """
    Parse the two-line ADDRESS: / MLS: response format.
    Returns (address, mls) with None for any NOT_FOUND value.
    """
    address: str | None = None
    mls:     str | None = None

    for line in (raw or "").strip().splitlines():
        line = line.strip()
        if line.upper().startswith("ADDRESS:"):
            val = line.split(":", 1)[1].strip()
            if val and val.upper() != "NOT_FOUND":
                address = val
        elif line.upper().startswith("MLS:"):
            val = line.split(":", 1)[1].strip()
            digits = re.sub(r"\D", "", val)
            if digits:
                mls = digits

    return address, mls


# ── Filename builder ────────────────────────────────────────────────────────────

def build_safe_filename(address: str | None, mls: str | None) -> str:
    """
    Build a filesystem-safe filename stem from address and MLS.

    Examples:
        ("123 Main St", "12345678")  →  "123_Main_St_MLS_12345678"
        ("188 Fairview Mall Dr", None) →  "188_Fairview_Mall_Dr"
        (None, "12345678")           →  "MLS_12345678"
        (None, None)                 →  "Unknown_Property"
    """
    parts: list[str] = []

    if address:
        safe = re.sub(r"[^\w\s\-]", "", address)   # strip punctuation
        safe = re.sub(r"\s+", "_", safe).strip("_") # spaces → underscores
        if safe:
            parts.append(safe)

    if mls:
        parts.append(f"MLS_{mls}")

    return "_".join(parts) if parts else "Unknown_Property"
