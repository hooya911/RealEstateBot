"""
processor.py — Audio conversion (ffmpeg) and transcription (OpenAI Whisper)

All audio operations use ffmpeg only — no pydub dependency.
ffmpeg is available on all platforms and is installed on Railway via nixpacks.toml.

  convert_to_mp3()        OGG/any → 192 kbps stereo MP3
  trim_audio_to_seconds() MP3 → first N seconds (ffmpeg stream copy, very fast)
  transcribe_audio()      MP3 → text via OpenAI Whisper API
"""
import os
import subprocess
import logging
from openai import OpenAI

from config import OPENAI_API_KEY, WHISPER_INITIAL_PROMPT

logger = logging.getLogger(__name__)

_openai = OpenAI(api_key=OPENAI_API_KEY)


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


# ── Step 2: Whisper transcription ─────────────────────────────────────────────

def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe an MP3 file via OpenAI Whisper API (whisper-1).

    The 2-minute trimmed clip is well under the 25 MB Whisper limit,
    so this is always a single direct API call.

    Args:
        audio_path: Path to the MP3 file.

    Returns:
        Transcription as a plain string.
    """
    size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    logger.info("Sending %.1f MB to Whisper...", size_mb)

    with open(audio_path, "rb") as f:
        response = _openai.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            prompt=WHISPER_INITIAL_PROMPT,
            response_format="verbose_json",
        )

    detected = getattr(response, "language", "unknown")
    logger.info("Whisper done — language: %s | chars: %d", detected, len(response.text))
    return response.text
