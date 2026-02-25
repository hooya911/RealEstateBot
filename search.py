"""
search.py — Property address lookup via Brave Search API

PRIMARY:  GPT-4o — primary for address parsing (better at bilingual context)
FALLBACK: Gemini — triggers automatically if GPT-4o is unavailable

TWO-PASS SEARCH STRATEGY:
  Pass 1 (MLS-based)     → finds the street address using the MLS number
  Pass 2 (address-based) → searches the address for detailed specs
                           (beds, baths, sqft, price, parking, etc.)

  Combining both passes gives a much richer listing_data block for the
  AI analyzer to use in the investment report.

Public API:
  find_property_address_with_data(mls)  → (address, listing_data_text)
  find_property_address(mls)            → address string only (backward-compat)
  sanitize_for_filename(address)        → safe filename stem
"""
import re
import logging
import httpx
from google import genai
from google.genai import types
from openai import OpenAI

from config import BRAVE_API_KEY, GOOGLE_API_KEY, OPENAI_API_KEY, GEMINI_MODEL

logger = logging.getLogger(__name__)

_gemini = genai.Client(api_key=GOOGLE_API_KEY)
_openai = OpenAI(api_key=OPENAI_API_KEY)

_BRAVE_URL = "https://api.search.brave.com/res/v1/web/search"

_BRAVE_HEADERS = {
    "Accept":               "application/json",
    "Accept-Encoding":      "gzip",
    "X-Subscription-Token": BRAVE_API_KEY,
}


# ── Brave Search helpers ────────────────────────────────────────────────────────

def _brave_get(query: str, count: int = 5) -> list[dict]:
    """Execute a single Brave Search query and return raw result list."""
    if not BRAVE_API_KEY:
        logger.warning("BRAVE_API_KEY not set — skipping web search.")
        return []
    try:
        with httpx.Client(timeout=12) as client:
            resp = client.get(
                _BRAVE_URL,
                headers=_BRAVE_HEADERS,
                params={"q": query, "count": count, "search_lang": "en", "country": "CA"},
            )
            resp.raise_for_status()
            results = resp.json().get("web", {}).get("results", [])
        logger.info("Brave [%r] → %d results", query[:60], len(results))
        return results
    except httpx.HTTPStatusError as exc:
        logger.error("Brave HTTP %s for query %r: %s", exc.response.status_code, query[:60], exc)
        return []
    except Exception as exc:
        logger.error("Brave search failed: %s", exc)
        return []


def _brave_search_by_mls(mls_number: str) -> list[dict]:
    """
    Pass 1 — Find listing by MLS number.
    Quotes around MLS number force exact-match on listing pages.
    Multiple query variants tried to maximise hit rate on Canadian MLS databases.
    """
    # Try exact-quoted MLS first (most specific)
    results = _brave_get(f'"{mls_number}" real estate listing Canada')
    if results:
        return results
    # Fallback: without quotes but with listing context
    return _brave_get(f"MLS {mls_number} property listing bedrooms bathrooms")


def _brave_search_by_address(address: str, mls_number: str) -> list[dict]:
    """
    Pass 2 — Find detailed specs using the address we already discovered.
    This query specifically targets beds/baths/sqft/price data in listing pages.
    """
    # Use address in quotes for precision
    results = _brave_get(f'"{address}" {mls_number} bedrooms bathrooms sqft price listing')
    if results:
        return results
    # Fallback: just address + listing
    return _brave_get(f'"{address}" real estate listing price')


# ── Listing data formatter ─────────────────────────────────────────────────────

def _format_listing_data(results: list[dict], label: str = "") -> str:
    """
    Convert raw Brave Search result dicts into a readable text block.
    Titles and snippets from listing pages usually contain beds/baths/sqft/price.
    """
    if not results:
        return ""

    header = f"=== {label} ===\n" if label else ""
    lines  = [header] if header else []
    for i, r in enumerate(results[:5], 1):
        title   = r.get("title",       "").strip()
        snippet = r.get("description", "").strip()
        url     = r.get("url",         "").strip()
        if title or snippet:
            if title:
                lines.append(f"• {title}")
            if snippet:
                lines.append(f"  {snippet}")
            if url:
                lines.append(f"  [{url}]")
            lines.append("")

    return "\n".join(lines).strip()


# ── Address extraction (GPT-4o primary, Gemini fallback) ───────────────────────

_ADDRESS_PROMPT = """You are a real estate data extraction assistant.

Below are web search results for the property with MLS number: {mls_number}

SEARCH RESULTS:
---
{results_text}
---

Task: Extract the STREET ADDRESS of this property.
Examples of valid output: "123 Main St", "456 Oak Ave Unit 3", "188 Fairview Mall Dr"

Rules:
- Return ONLY the street address — no city, province, country, or postal code.
- Do NOT add explanation or punctuation after the address.
- If no clear address is found, reply with exactly: NOT_FOUND

Street address:"""


def _extract_address_with_ai(mls_number: str, results: list[dict]) -> str | None:
    """
    Parse search results to extract the street address.
    GPT-4o is primary (better contextual reasoning); Gemini is fallback.
    """
    if not results:
        return None

    results_text = "\n\n".join(
        f"Result {i+1}:\n  Title: {r.get('title', '')}\n"
        f"  URL: {r.get('url', '')}\n  Snippet: {r.get('description', '')}"
        for i, r in enumerate(results[:5])
    )
    prompt = _ADDRESS_PROMPT.format(mls_number=mls_number, results_text=results_text)

    # ── Primary: GPT-4o ────────────────────────────────────────────────────────
    try:
        response = _openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=60,
        )
        raw = response.choices[0].message.content.strip()
        logger.info("GPT-4o address: %r", raw)
        return None if (not raw or raw.upper() == "NOT_FOUND") else raw

    except Exception as exc:
        logger.warning("GPT-4o address extraction failed: %s — switching to Gemini", exc)

    # ── Fallback: Gemini ───────────────────────────────────────────────────────
    try:
        response = _gemini.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=60),
        )
        raw = response.text.strip()
        logger.info("Gemini address (fallback): %r", raw)
        return None if (not raw or raw.upper() == "NOT_FOUND") else raw

    except Exception as exc:
        logger.error("Gemini address extraction also failed: %s", exc)
        return None


# ── Public API ─────────────────────────────────────────────────────────────────

def find_property_address_with_data(mls_number: str) -> tuple[str, str]:
    """
    Two-pass search returning (address, listing_data_text).

    Pass 1 — MLS-based search:
      Finds the street address. Also grabs any snippet data from that search.

    Pass 2 — Address-based search (runs only if Pass 1 found an address):
      Re-queries Brave with the address + MLS to pull beds/baths/sqft/price
      from listing pages that might not rank well for the bare MLS number.

    listing_data_text: merged results from both passes, ready for AI consumption.
    """
    # Pass 1: MLS → address + initial snippets
    mls_results  = _brave_search_by_mls(mls_number)
    mls_snippets = _format_listing_data(mls_results, label="MLS Search Results")
    address      = _extract_address_with_ai(mls_number, mls_results)

    # Pass 2: address → detailed listing specs (beds/baths/sqft/price)
    addr_snippets = ""
    if address:
        addr_results  = _brave_search_by_address(address, mls_number)
        addr_snippets = _format_listing_data(addr_results, label="Address Detail Search")
        logger.info("Pass 2 (address search) returned %d results", len(addr_results))

    # Merge both passes into one listing_data block
    parts        = [p for p in [mls_snippets, addr_snippets] if p]
    listing_data = "\n\n".join(parts) if parts else "No web listing data found."

    if address:
        logger.info("Property address resolved: %r", address)
        return address, listing_data

    fallback = f"MLS_{mls_number}"
    logger.warning("Address not found — using fallback: %s", fallback)
    return fallback, listing_data


def find_property_address(mls_number: str) -> str:
    """Backward-compatible wrapper — returns address string only."""
    address, _ = find_property_address_with_data(mls_number)
    return address


def fetch_listing_data(address: str | None, mls: str | None) -> str:
    """
    Fetch property listing details (beds, baths, sqft, price, parking, etc.)
    from Brave Search using the address and MLS number we already know.

    Unlike find_property_address_with_data() this skips Pass 1 (MLS→address)
    because the address was already extracted from the audio transcript.

    Strategy:
      - address + MLS  → address+MLS combined search (most precise)
      - address only   → address listing search
      - MLS only       → MLS listing search
      - neither        → returns empty string

    Returns:
        Formatted listing data text ready for AI consumption, or "" if nothing found.
    """
    if not BRAVE_API_KEY:
        logger.warning("BRAVE_API_KEY not set — skipping property detail lookup.")
        return ""

    results: list[dict] = []

    if address and mls:
        results = _brave_search_by_address(address, mls)
        if not results:
            # Fallback: try broader address-only search
            results = _brave_get(f'"{address}" real estate listing beds baths sqft price')
    elif address:
        results = _brave_get(f'"{address}" real estate listing beds baths sqft price')
    elif mls:
        results = _brave_search_by_mls(mls)

    if not results:
        logger.info("Brave fetch_listing_data: no results for address=%r mls=%r", address, mls)
        return ""

    logger.info("Brave fetch_listing_data: %d results for address=%r mls=%r", len(results), address, mls)
    return _format_listing_data(results, label="Property Listing Details from Web")


def sanitize_for_filename(address: str) -> str:
    """
    Convert an address string into a safe filename stem.
    "123 Main St, Vancouver"  →  "123_Main_St"
    "MLS_12345678"            →  "MLS_12345678"
    """
    street = address.split(",")[0].strip()
    safe   = re.sub(r"[^\w\s\-]", "", street)
    safe   = re.sub(r"\s+", "_", safe).strip("_")
    return safe or "Unknown_Property"
