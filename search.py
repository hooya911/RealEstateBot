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
import html as _html
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


# ── Listing page fetcher ───────────────────────────────────────────────────────

# Canadian real estate listing domains — pages from these are worth fetching
_LISTING_DOMAINS = {
    "realtor.ca", "housesigma.com", "zolo.ca", "rew.ca",
    "remax.ca", "royallepage.ca", "century21.ca", "coldwellbanker.ca",
    "sutton.com", "point2homes.com", "zoocasa.com", "condos.ca",
    "kijiji.ca", "propertyguys.com", "exp.com", "sothebysrealty.com",
}

# Browser-like headers so listing sites don't block the request
_PAGE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,*/*;q=0.9",
    "Accept-Language": "en-CA,en;q=0.9",
}


def _is_likely_listing_page(url: str) -> bool:
    """Return True if the URL looks like a real estate listing detail page."""
    for domain in _LISTING_DOMAINS:
        if domain in url:
            return True
    # Generic URL patterns used by broker sites
    return bool(re.search(
        r"/(listing|property|mls|real-estate|house|home|condo|property-detail)/",
        url, re.IGNORECASE,
    ))


def _strip_html(html_text: str) -> str:
    """
    Remove HTML tags, scripts, and styles; decode entities; collapse whitespace.
    Returns clean, readable text.
    """
    # Drop <script> and <style> blocks entirely
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html_text,
                  flags=re.DOTALL | re.IGNORECASE)
    # Drop HTML comments
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    # Replace block-level tags with newlines for readability
    text = re.sub(r"<(?:br|p|div|li|h[1-6]|tr|td|th)[^>]*>", "\n", text,
                  flags=re.IGNORECASE)
    # Strip remaining tags
    text = re.sub(r"<[^>]+>", " ", text)
    # Decode HTML entities (&amp; → & etc.)
    text = _html.unescape(text)
    # Collapse repeated whitespace / blank lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _fetch_page_text(url: str, max_chars: int = 8000) -> str:
    """
    Fetch a real estate listing page and return its stripped text content.

    Limits output to max_chars so we don't flood the AI context.
    Returns "" on any network/parse error.
    """
    try:
        with httpx.Client(timeout=15, follow_redirects=True,
                          headers=_PAGE_HEADERS) as client:
            resp = client.get(url)
            resp.raise_for_status()
            ctype = resp.headers.get("content-type", "")
            if "html" not in ctype:
                logger.info("Skipping non-HTML page: %s (%s)", url[:80], ctype)
                return ""
            text = _strip_html(resp.text)
            if len(text) > max_chars:
                text = text[:max_chars] + "\n[... page truncated ...]"
            logger.info("Fetched listing page %s — %d chars", url[:80], len(text))
            return text
    except Exception as exc:
        logger.warning("Could not fetch listing page %s: %s", url[:80], exc)
        return ""


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
    Fetch comprehensive property listing details from Brave Search.

    TWO-LAYER approach:
      Layer 1 — Brave snippets:
        Quick titles + short descriptions from the top search results.
        Usually contains: beds, baths, price, sqft at a glance.

      Layer 2 — Full page fetch:
        Visits the top 1-2 listing pages (realtor.ca, housesigma, zolo, etc.)
        and extracts the full visible text, which typically includes:
          • Size (sq ft), Lot Size, Basement type, Days on Market, Tax
          • Complete broker/marketing description (bullet points, features)
          • Parking, heating, garage, school zones, etc.

    The combined output gives Gemini everything it needs to write a rich
    Property Specs block at the top of the summary.

    Returns "" if BRAVE_API_KEY is not set or nothing is found.
    """
    if not BRAVE_API_KEY:
        logger.warning("BRAVE_API_KEY not set — skipping property detail lookup.")
        return ""

    # ── Layer 1: Brave Search snippets ────────────────────────────────────────
    results: list[dict] = []
    if address and mls:
        results = _brave_search_by_address(address, mls)
        if not results:
            results = _brave_get(f'"{address}" real estate listing beds baths sqft price')
    elif address:
        results = _brave_get(f'"{address}" real estate listing beds baths sqft price')
    elif mls:
        results = _brave_search_by_mls(mls)

    if not results:
        logger.info("Brave fetch_listing_data: no results for address=%r mls=%r", address, mls)
        return ""

    logger.info("Brave fetch_listing_data: %d results for address=%r mls=%r",
                len(results), address, mls)

    snippets_text = _format_listing_data(results, label="Brave Search Snippets")

    # ── Layer 2: Full listing page fetch ──────────────────────────────────────
    # Try up to 3 result URLs; take the first two that are real listing pages
    page_texts: list[str] = []
    for result in results[:5]:
        if len(page_texts) >= 2:
            break
        url = result.get("url", "").strip()
        if not url or not _is_likely_listing_page(url):
            continue
        page_text = _fetch_page_text(url, max_chars=8000)
        if page_text and len(page_text) > 300:
            source_label = url.split("/")[2]  # e.g. "www.realtor.ca"
            page_texts.append(
                f"=== Full Listing Page: {source_label} ===\n{page_text}"
            )

    # ── Combine layers ────────────────────────────────────────────────────────
    parts = [p for p in [snippets_text] + page_texts if p]
    combined = "\n\n".join(parts)
    logger.info("fetch_listing_data complete — %d chars total", len(combined))
    return combined


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
