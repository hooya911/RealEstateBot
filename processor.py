"""
processor.py — Audio conversion (ffmpeg), trimming (pydub), and transcription (OpenAI Whisper)

Two-step audio pipeline:
  Step 1 — convert_to_mp3():
      Takes the raw OGG (or any format) file received from Telegram and
      converts it to a 192 kbps stereo MP3 using ffmpeg.
      Command: ffmpeg -i input.ogg -vn -ar 44100 -ac 2 -b:a 192k output.mp3

  Step 2 — transcribe_audio():
      • Files ≤ 24 MB → sent directly to Whisper (single API call).
      • Files  > 24 MB → automatically split into timed chunks via pydub,
        each chunk transcribed separately, results joined into one transcript.
      Language is NOT forced so Whisper auto-detects bilingual Farsi/English.
"""
import os
import math
import tempfile
import subprocess
import logging
from openai import OpenAI
from pydub import AudioSegment

from config import OPENAI_API_KEY, WHISPER_INITIAL_PROMPT

logger = logging.getLogger(__name__)

_openai = OpenAI(api_key=OPENAI_API_KEY)

_WHISPER_SAFE_MB = 24.0   # 1 MB below the hard 25 MB Whisper API limit


# ── Step 1: OGG → MP3 conversion ──────────────────────────────────────────────

def convert_to_mp3(input_path: str, output_path: str | None = None) -> str:
    """
    Convert any audio file to a high-quality 192 kbps stereo MP3 via ffmpeg.

    This normalises whatever format Telegram delivers (OGG Opus, M4A, etc.)
    into a consistent format that Whisper handles best.

    ffmpeg command used (as specified):
        ffmpeg -i input.ogg -vn -ar 44100 -ac 2 -b:a 192k output.mp3

    Args:
        input_path:  Path to the source audio file.
        output_path: Destination path for the MP3.
                     Defaults to same directory with the extension replaced.

    Returns:
        Absolute path to the created MP3 file.

    Raises:
        RuntimeError  if ffmpeg is not found or conversion fails.
    """
    if output_path is None:
        base        = os.path.splitext(input_path)[0]
        output_path = f"{base}.mp3"

    cmd = [
        "ffmpeg",
        "-i",    input_path,
        "-vn",              # strip video stream (safe to use on pure audio too)
        "-ar",   "44100",   # 44.1 kHz sample rate
        "-ac",   "2",       # stereo
        "-b:a",  "192k",    # 192 kbps constant bitrate
        "-y",               # overwrite output without prompting
        output_path,
    ]

    logger.info("Converting audio: %s → %s", input_path, output_path)
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg failed to convert the audio file.\n"
            "Make sure ffmpeg is installed and on your PATH.\n"
            f"ffmpeg stderr: {result.stderr[-600:]}"
        )

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    logger.info("Conversion complete → %s (%.1f MB)", output_path, size_mb)
    return output_path


# ── Step 1b: Trim audio to N seconds ──────────────────────────────────────────

def trim_audio_to_seconds(audio_path: str, seconds: int = 120) -> str:
    """
    Trim an MP3 to the first N seconds using pydub.
    Returns the path to a NEW trimmed file — the original is untouched.
    Caller is responsible for deleting the trimmed file when done.

    Args:
        audio_path:  Path to the source MP3.
        seconds:     Maximum duration to keep (default 120 = 2 minutes).

    Returns:
        Path to the trimmed MP3 (same directory, suffix _trimmed.mp3).
    """
    audio        = AudioSegment.from_file(audio_path, format="mp3")
    trimmed      = audio[: seconds * 1000]          # pydub uses milliseconds
    base         = os.path.splitext(audio_path)[0]
    trimmed_path = f"{base}_trimmed.mp3"
    trimmed.export(trimmed_path, format="mp3", bitrate="192k")
    logger.info(
        "Trimmed to %ds → %s (original %.1fs)",
        seconds,
        os.path.basename(trimmed_path),
        len(audio) / 1000,
    )
    return trimmed_path


# ── Whisper helpers ────────────────────────────────────────────────────────────

def _transcribe_single(audio_path: str) -> str:
    """
    Send one audio file (must be ≤ 25 MB) to OpenAI Whisper and return text.
    Language auto-detected; prompt biases toward bilingual real-estate vocab.
    """
    with open(audio_path, "rb") as audio_file:
        response = _openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            # No `language` param → auto-detect for natural bilingual output
            prompt=WHISPER_INITIAL_PROMPT,   # correct param name (not initial_prompt)
            response_format="verbose_json",  # exposes .language & .segments
        )

    detected = getattr(response, "language", "unknown")
    logger.info(
        "Chunk transcribed — language: %s | chars: %d",
        detected, len(response.text),
    )
    return response.text


def _split_and_transcribe(audio_path: str) -> str:
    """
    Split an oversized MP3 into safe-sized chunks using pydub, transcribe
    each chunk with Whisper, and return the joined full transcript.

    The split is purely time-based: the total duration is divided evenly into
    as many equal segments as needed so each exported segment is ≤ 24 MB.
    """
    size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    logger.info("Audio is %.1f MB — loading with pydub for chunking", size_mb)

    audio    = AudioSegment.from_file(audio_path, format="mp3")
    total_ms = len(audio)                          # total duration in milliseconds

    # Minimum number of chunks so each is ≤ _WHISPER_SAFE_MB
    n_chunks = math.ceil(size_mb / _WHISPER_SAFE_MB)
    chunk_ms = math.ceil(total_ms / n_chunks)      # duration per chunk in ms

    logger.info(
        "Splitting into %d chunks (~%d s each)",
        n_chunks, chunk_ms // 1000,
    )

    transcripts: list[str] = []
    chunk_dir = tempfile.mkdtemp(prefix="re_bot_chunks_")

    try:
        for i in range(n_chunks):
            start = i * chunk_ms
            end   = min(start + chunk_ms, total_ms)
            chunk = audio[start:end]

            chunk_path = os.path.join(chunk_dir, f"chunk_{i:03d}.mp3")
            chunk.export(chunk_path, format="mp3", bitrate="192k")

            chunk_mb = os.path.getsize(chunk_path) / (1024 * 1024)
            logger.info(
                "Transcribing chunk %d/%d — %.1f MB (%ds – %ds)",
                i + 1, n_chunks, chunk_mb, start // 1000, end // 1000,
            )

            text = _transcribe_single(chunk_path)
            transcripts.append(text)

    finally:
        # Always clean up temp chunk files
        for fname in os.listdir(chunk_dir):
            try:
                os.remove(os.path.join(chunk_dir, fname))
            except OSError:
                pass
        try:
            os.rmdir(chunk_dir)
        except OSError:
            pass

    full_transcript = " ".join(transcripts)
    logger.info(
        "All %d chunks joined — total transcript: %d chars",
        n_chunks, len(full_transcript),
    )
    return full_transcript


# ── Step 2: Whisper transcription (public API) ─────────────────────────────────

def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe an MP3 file via OpenAI Whisper API (whisper-1).

    Automatically handles files of any size:
      • ≤ 24 MB → single direct Whisper call
      • > 24 MB → pydub splits into equal time chunks, each transcribed
                  separately, results concatenated into one full transcript

    Args:
        audio_path: Path to the MP3 file (output of convert_to_mp3).

    Returns:
        Full transcription as a plain string.

    Raises:
        openai.APIError  on any API-level failure.
    """
    size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    logger.info("Audio file: %.1f MB → %s", size_mb, audio_path)

    if size_mb <= _WHISPER_SAFE_MB:
        logger.info("File is within limit — sending directly to Whisper")
        return _transcribe_single(audio_path)
    else:
        logger.info("File is %.1f MB — auto-chunking for Whisper (limit 24 MB)", size_mb)
        return _split_and_transcribe(audio_path)
