"""
processor.py — Audio conversion (ffmpeg) and transcription (Google Cloud Chirp 3)

All audio operations use ffmpeg only — no pydub dependency.
ffmpeg is available on all platforms and is installed on Railway via nixpacks.toml.

  convert_to_mp3()        OGG/any → 192 kbps stereo MP3
  trim_audio_to_seconds() MP3 → first N seconds (ffmpeg stream copy, very fast)
  transcribe_audio()      MP3 → text via Google Cloud Speech-to-Text Chirp 3
                          (async, 55-second ffmpeg chunks — no memory limit)
"""
import os
import re
import json
import math
import asyncio
import tempfile
import subprocess
import logging

from config import GOOGLE_CLOUD_CREDENTIALS

logger = logging.getLogger(__name__)


def get_audio_duration_secs(path: str) -> float:
    """Return audio duration in seconds using ffprobe (no file load into RAM)."""
    # Try stream-level duration first
    probe = subprocess.run(
        ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams', path],
        capture_output=True, text=True
    )
    try:
        for stream in json.loads(probe.stdout).get('streams', []):
            if 'duration' in stream:
                return float(stream['duration'])
    except Exception:
        pass

    # Fallback: format-level duration
    probe2 = subprocess.run(
        ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', path],
        capture_output=True, text=True
    )
    try:
        return float(json.loads(probe2.stdout).get('format', {}).get('duration', 0))
    except Exception:
        return 0


# ── Step 1a: OGG → MP3 conversion ─────────────────────────────────────────────

def convert_to_mp3(input_path: str, output_path: str | None = None) -> str:
    """
    Convert any audio file to a high-quality 192 kbps stereo MP3 via ffmpeg.

    ffmpeg command: ffmpeg -i input -vn -ar 44100 -ac 2 -b:a 192k output.mp3

    Args:
        input_path:  Path to the source audio file.
        output_path: Destination MP3 path. Defaults to same name with .mp3 extension.

    Returns:
        Absolute path to the created MP3 file.

    Raises:
        RuntimeError if ffmpeg fails.
    """
    if output_path is None:
        output_path = f"{os.path.splitext(input_path)[0]}.mp3"

    cmd = [
        "ffmpeg",
        "-i",   input_path,
        "-vn",              # strip any video stream
        "-ar",  "44100",    # 44.1 kHz sample rate
        "-ac",  "2",        # stereo
        "-b:a", "192k",     # 192 kbps constant bitrate
        "-y",               # overwrite without prompting
        output_path,
    ]

    logger.info("Converting: %s → %s", input_path, output_path)
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed to convert audio.\n"
            "Make sure ffmpeg is installed and on your PATH.\n"
            f"stderr: {result.stderr[-600:]}"
        )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info("Conversion done → %.1f MB", size_mb)
    return output_path


# ── Step 1b: Trim to first N seconds ──────────────────────────────────────────

def trim_audio_to_seconds(audio_path: str, seconds: int = 120) -> str:
    """
    Trim an MP3 to the first N seconds using ffmpeg stream copy.

    Uses '-c:a copy' (no re-encoding) so this completes almost instantly
    regardless of file size. No pydub or audioop required.

    Args:
        audio_path:  Path to the source MP3.
        seconds:     Maximum duration to keep (default 120 = 2 minutes).

    Returns:
        Path to the trimmed MP3 (same directory, suffix _trimmed.mp3).
        Caller is responsible for deleting this file when done.

    Raises:
        RuntimeError if ffmpeg fails.
    """
    base         = os.path.splitext(audio_path)[0]
    trimmed_path = f"{base}_trimmed.mp3"

    cmd = [
        "ffmpeg",
        "-i",  audio_path,
        "-t",  str(seconds),  # stop after N seconds
        "-c:a", "copy",        # stream copy — no re-encode, instant
        "-y",
        trimmed_path,
    ]

    logger.info("Trimming to %ds → %s", seconds, os.path.basename(trimmed_path))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to trim audio.\nstderr: {result.stderr[-400:]}"
        )

    size_mb = os.path.getsize(trimmed_path) / (1024 * 1024)
    logger.info("Trim done → %.1f MB", size_mb)
    return trimmed_path


# ── Anti-repetition guard for Chirp 3 ASR hallucinations ──────────────────────

def _collapse_runaway_repeats(text: str, max_repeats: int = 3) -> str:
    """
    Collapse consecutive phrase repetitions caused by Chirp 3 ASR hallucinations.

    Chirp 3 occasionally gets stuck in a loop on low-content / silent / noisy
    audio and emits the same short phrase hundreds of times in a row
    (e.g. "این چیز، این چیز، این چیز، ...").  This walks the token stream
    left-to-right; whenever the same n-gram (n = 1..6 tokens) repeats more
    than `max_repeats` times in a row, only the first `max_repeats` copies
    are kept and the rest are discarded.

    Also collapses 5+ identical chars in a single token down to 3 (catches
    single-syllable loops like "آآآآآآآآ").

    Works for any whitespace-separated script (English, Persian/Arabic, etc.).
    """
    if not text:
        return text

    # Char-level guard: collapse 5+ identical chars in a row down to 3.
    text = re.sub(r"(.)\1{4,}", lambda m: m.group(1) * 3, text)

    tokens = text.split()
    if len(tokens) < 8:
        return text

    out: list[str] = []
    i = 0
    n = len(tokens)
    while i < n:
        collapsed = False
        # Try larger n-grams first so multi-word loops get caught.
        for size in range(min(6, (n - i) // 2), 0, -1):
            phrase = tokens[i:i + size]
            count = 1
            j = i + size
            while j + size <= n and tokens[j:j + size] == phrase:
                count += 1
                j += size
            if count > max_repeats:
                out.extend(phrase * max_repeats)
                i = j
                collapsed = True
                break
        if not collapsed:
            out.append(tokens[i])
            i += 1
    return " ".join(out)

# ── Step 2: Chirp 3 transcription ─────────────────────────────────────────────

async def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe an MP3 file via Google Cloud Speech-to-Text Chirp 3.

    Each 55-second chunk is extracted via ffmpeg directly from disk — the full
    audio file is never loaded into RAM, keeping memory usage constant regardless
    of recording length.

    Language codes: Farsi (fa-IR) + English (en-US) — same bilingual setup as
    the telegram-audio-bot, matching the real estate walkthrough context.

    Args:
        audio_path: Path to the MP3 file.

    Returns:
        Transcription as a plain string.

    Raises:
        ValueError if GOOGLE_CLOUD_CREDENTIALS is not set.
    """
    from google.cloud.speech_v2 import SpeechClient
    from google.cloud.speech_v2.types import cloud_speech
    from google.api_core.client_options import ClientOptions
    from google.oauth2 import service_account

    if not GOOGLE_CLOUD_CREDENTIALS:
        raise ValueError("GOOGLE_CLOUD_CREDENTIALS environment variable is not set")

    credentials_info = json.loads(GOOGLE_CLOUD_CREDENTIALS)
    project_id = credentials_info.get('project_id')
    if not project_id:
        raise ValueError("project_id not found in GOOGLE_CLOUD_CREDENTIALS")

    credentials = service_account.Credentials.from_service_account_info(
        credentials_info,
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )

    client_options = ClientOptions(api_endpoint="us-speech.googleapis.com")
    client = SpeechClient(credentials=credentials, client_options=client_options)

    config = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        model="chirp_3",
        language_codes=["fa-IR", "en-US"],
        features=cloud_speech.RecognitionFeatures(
            enable_automatic_punctuation=True,
        ),
    )
    recognizer_path = f"projects/{project_id}/locations/us/recognizers/_"

    duration_secs = get_audio_duration_secs(audio_path)
    CHUNK_SECS = 55
    num_chunks = max(1, math.ceil(duration_secs / CHUNK_SECS))
    logger.info("Audio: %.1fs — %d chunk(s) for Chirp 3", duration_secs, num_chunks)

    loop = asyncio.get_running_loop()
    all_lines: list[str] = []

    for idx in range(num_chunks):
        chunk_offset_secs = idx * CHUNK_SECS

        # Extract this 55-second window with ffmpeg — reads only that slice
        tmp = tempfile.NamedTemporaryFile(suffix='.mp3', delete=False)
        tmp.close()
        try:
            result = subprocess.run(
                [
                    'ffmpeg', '-y',
                    '-ss', str(chunk_offset_secs),   # fast seek before -i
                    '-t', str(CHUNK_SECS),
                    '-i', audio_path,
                    '-ar', '16000', '-ac', '1', '-b:a', '16k',
                    '-f', 'mp3', tmp.name
                ],
                capture_output=True
            )
            if result.returncode != 0 or os.path.getsize(tmp.name) < 500:
                logger.info("Chunk %d/%d: empty or ffmpeg error — skipping", idx + 1, num_chunks)
                continue

            with open(tmp.name, 'rb') as f:
                chunk_bytes = f.read()
        finally:
            if os.path.exists(tmp.name):
                os.unlink(tmp.name)

        request = cloud_speech.RecognizeRequest(
            recognizer=recognizer_path,
            config=config,
            content=chunk_bytes,
        )

        try:
            response = await loop.run_in_executor(
                None, lambda r=request: client.recognize(request=r)
            )
        except Exception as chunk_err:
            logger.error("Chunk %d/%d failed: %s", idx + 1, num_chunks, chunk_err)
            continue

        chunk_pieces: list[str] = []
        for res in response.results:
            if not res.alternatives:
                continue
            text = res.alternatives[0].transcript.strip()
            if text:
                chunk_pieces.append(text)

        if chunk_pieces:
            chunk_text = " ".join(chunk_pieces)
            chunk_text_clean = _collapse_runaway_repeats(chunk_text)
            if chunk_text != chunk_text_clean:
                logger.info(
                    "Chunk %d/%d: collapsed runaway repeats (%d → %d chars)",
                    idx + 1, num_chunks, len(chunk_text), len(chunk_text_clean),
                )
            all_lines.append(chunk_text_clean)

        logger.info("Chunk %d/%d done", idx + 1, num_chunks)

    if not all_lines:
        return "No speech detected in audio"
    joined = "\n".join(all_lines)
    # Final pass catches loops that span chunk boundaries.
    return _collapse_runaway_repeats(joined)
