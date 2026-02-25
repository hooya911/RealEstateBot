"""
bot.py — Real Estate File Renamer Bot
Run with:  python bot.py

─── WHAT IT DOES ────────────────────────────────────────────────────────────────
Forward TWO files (any order — simultaneously is fine):
    🎙️ The OGG audio file
    📹 The video file  (ANY size — never downloaded)

Pipeline (triggered when audio arrives):
  1. Download OGG → convert to 192 kbps MP3  (full quality, full length)
  2. Trim a 2-minute copy for analysis        (original MP3 unchanged)
  3. Chirp 3 transcribes the 2-min clip       (Google Cloud Speech-to-Text)
  4. GPT-4o extracts: Property Address + MLS number
  5. Delete the 2-min clip
  6. Rename the full MP3 → "{Address}_MLS_{Number}.mp3"

Delivery (fires when both files are ready):
  7. Re-send video via Telegram file_id — caption = new filename
     (no download, works for any size)
  8. Send renamed MP3
  9. Chirp 3 transcribes the full MP3         (55-sec ffmpeg chunks, no RAM limit)
 10. Send full transcription
 11. Gemini generates AI summary → send summary

─── STATE ───────────────────────────────────────────────────────────────────────
  _VIDEO_KEY       → file_id + type (stored, never downloaded)
  _AUDIO_PROCESSED → safe_name, audio_filename, mp3_path
─────────────────────────────────────────────────────────────────────────────────
"""
import os
import html
import asyncio
import logging

from telegram import Update, InputFile
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

from config import TELEGRAM_BOT_TOKEN, ALLOWED_USER_IDS, TEMP_DIR, TELEGRAM_MAX_FILE_MB, GOOGLE_API_KEY
from processor import convert_to_mp3, trim_audio_to_seconds, transcribe_audio, get_audio_duration_secs
from analyzer import extract_address_and_mls, build_safe_filename

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

os.makedirs(TEMP_DIR, exist_ok=True)

# ── State keys ─────────────────────────────────────────────────────────────────
_VIDEO_KEY       = "pending_video"
_AUDIO_PROCESSED = "audio_processed"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _is_authorized(user_id: int) -> bool:
    if not ALLOWED_USER_IDS:
        return True
    return user_id in ALLOWED_USER_IDS


def _e(value) -> str:
    """Escape value for safe inclusion in ParseMode.HTML messages."""
    return html.escape(str(value))


def _cleanup(*paths: str | None) -> None:
    for path in paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
                logger.info("Cleaned up: %s", path)
            except OSError as exc:
                logger.warning("Could not delete %s: %s", path, exc)


# ── Commands ───────────────────────────────────────────────────────────────────

async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "🏠 <b>Real Estate File Renamer Bot</b>\n\n"
        "Forward <b>two files</b> — in any order, even at the same time:\n\n"
        "1️⃣  🎙️  <b>Audio file</b> (OGG)\n"
        "2️⃣  📹  <b>Video file</b> (any size — no limit)\n\n"
        "📦 <b>You'll get back:</b>\n"
        "• 📹 Video re-sent with the property name as caption\n"
        "• 🎵 Audio renamed to the property address + MLS\n\n"
        "Commands: /status  /reset",
        parse_mode=ParseMode.HTML,
    )


async def cmd_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    has_v = _VIDEO_KEY       in context.user_data
    has_a = _AUDIO_PROCESSED in context.user_data
    v     = "✅ Video stored"    if has_v else "⏳ Waiting for video"
    a     = "✅ Audio processed" if has_a else "⏳ Waiting for audio"
    info  = context.user_data.get(_AUDIO_PROCESSED, {})
    msg   = f"<b>Status:</b>\n{v}\n{a}"
    if info:
        msg += f"\n\n🏷 <code>{_e(info.get('safe_name', '?'))}</code>"
    await update.message.reply_text(msg, parse_mode=ParseMode.HTML)


async def cmd_reset(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    audio_info = context.user_data.pop(_AUDIO_PROCESSED, None)
    context.user_data.pop(_VIDEO_KEY, None)
    if audio_info:
        _cleanup(audio_info.get("mp3_path"))
    await update.message.reply_text(
        "🔄 <b>Reset.</b> Ready for the next property.",
        parse_mode=ParseMode.HTML,
    )


# ── Video handler — stores file_id only, NEVER downloads ──────────────────────

async def handle_video(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if not _is_authorized(message.from_user.id):
        await message.reply_text("⛔ Not authorized.")
        return

    is_document = message.document is not None
    file_obj    = message.video or message.document
    if file_obj is None:
        return

    size_mb   = (file_obj.file_size or 0) / (1024 * 1024)
    orig_name = getattr(file_obj, "file_name", None) or f"{file_obj.file_id}.mp4"

    context.user_data[_VIDEO_KEY] = {
        "file_id":     file_obj.file_id,
        "is_document": is_document,
        "orig_name":   orig_name,
        "size_mb":     round(size_mb, 1),
    }
    logger.info("Video stored (ref only): %s  %.1f MB  doc=%s", orig_name, size_mb, is_document)

    if _AUDIO_PROCESSED in context.user_data:
        await message.reply_text(
            "📹 <b>Video received!</b> Audio already done — delivering now...",
            parse_mode=ParseMode.HTML,
        )
        await _deliver_package(update, context)
    else:
        await message.reply_text(
            f"✅ <b>Video stored</b> (<code>{_e(orig_name)}</code>, {size_mb:.1f} MB).\n"
            "Now forward the <b>audio file</b> (OGG).",
            parse_mode=ParseMode.HTML,
        )


# ── Audio handler — full pipeline ─────────────────────────────────────────────

async def handle_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    user    = message.from_user

    if not _is_authorized(user.id):
        await message.reply_text("⛔ Not authorized.")
        return

    # Resolve file object
    if message.audio:
        file_obj  = message.audio
        orig_name = message.audio.file_name or f"{file_obj.file_id}.ogg"
    elif message.voice:
        file_obj  = message.voice
        orig_name = f"{file_obj.file_id}.ogg"
    elif message.document:
        file_obj  = message.document
        orig_name = file_obj.file_name or f"{file_obj.file_id}.ogg"
    else:
        return

    size_mb = (file_obj.file_size or 0) / (1024 * 1024)
    if size_mb > TELEGRAM_MAX_FILE_MB:
        await message.reply_text(
            f"⚠️ Audio is {size_mb:.1f} MB — Telegram bot limit is {TELEGRAM_MAX_FILE_MB} MB.\n"
            "Please trim the recording and try again."
        )
        return

    raw_path:     str | None = os.path.join(TEMP_DIR, f"{file_obj.file_id}_raw_{orig_name}")
    mp3_path:     str | None = os.path.join(TEMP_DIR, f"{file_obj.file_id}.mp3")
    trimmed_path: str | None = None

    try:
        # ── 1. Download ────────────────────────────────────────────────────────
        await message.reply_text(
            "🎙️ <b>Audio received! Downloading...</b>", parse_mode=ParseMode.HTML
        )
        await context.bot.send_chat_action(chat_id=message.chat_id, action=ChatAction.TYPING)
        tg_file = await context.bot.get_file(file_obj.file_id)
        await tg_file.download_to_drive(raw_path)

        # ── 2. Convert to full-quality MP3 ────────────────────────────────────
        await message.reply_text(
            "🔄 <b>Converting to 192 kbps MP3...</b>", parse_mode=ParseMode.HTML
        )
        mp3_path = convert_to_mp3(raw_path, mp3_path)
        _cleanup(raw_path)
        raw_path = None

        # ── 3. Trim to first 2 minutes for analysis ───────────────────────────
        await message.reply_text(
            "✂️ <b>Trimming to first 2 minutes for address extraction...</b>",
            parse_mode=ParseMode.HTML,
        )
        trimmed_path = trim_audio_to_seconds(mp3_path, seconds=120)

        # ── 4. Transcribe the 2-min clip ──────────────────────────────────────
        await message.reply_text(
            "📝 <b>Transcribing first 2 minutes...</b>", parse_mode=ParseMode.HTML
        )
        transcript = await transcribe_audio(trimmed_path)
        _cleanup(trimmed_path)
        trimmed_path = None

        # ── 5. GPT-4o: extract address + MLS ─────────────────────────────────
        await message.reply_text(
            "🔍 <b>Extracting address and MLS number...</b>", parse_mode=ParseMode.HTML
        )
        address, mls = extract_address_and_mls(transcript)
        safe_name    = build_safe_filename(address, mls)
        logger.info("Extracted → address=%r  mls=%r  safe_name=%r", address, mls, safe_name)

        # ── 6. Rename full MP3 ────────────────────────────────────────────────
        audio_filename = f"{safe_name}.mp3"
        named_mp3      = os.path.join(TEMP_DIR, audio_filename)
        if os.path.exists(named_mp3):
            named_mp3      = os.path.join(TEMP_DIR, f"{safe_name}_{file_obj.file_id}.mp3")
            audio_filename = os.path.basename(named_mp3)
        os.rename(mp3_path, named_mp3)
        mp3_path = named_mp3

        # ── 7. Store state ────────────────────────────────────────────────────
        context.user_data[_AUDIO_PROCESSED] = {
            "safe_name":      safe_name,
            "audio_filename": audio_filename,
            "mp3_path":       mp3_path,
        }

        await message.reply_text(
            f"✅ <b>Audio ready!</b>\n\n"
            f"🏷  <code>{_e(safe_name)}</code>\n\n"
            f"{'📍 Address : <code>' + _e(address) + '</code>' if address else '⚠️ Address not found in first 2 minutes'}\n"
            f"{'🔑 MLS     : <code>' + _e(mls) + '</code>' if mls else '⚠️ MLS not found in first 2 minutes'}",
            parse_mode=ParseMode.HTML,
        )

        # Deliver immediately if video already stored
        if _VIDEO_KEY in context.user_data:
            await message.reply_text(
                "📹 <b>Video already received — delivering now!</b>",
                parse_mode=ParseMode.HTML,
            )
            await _deliver_package(update, context)
        else:
            await message.reply_text(
                "📹 <b>Now forward the video file</b> to complete delivery.\n"
                "<i>Any size is fine — I won't download it.</i>",
                parse_mode=ParseMode.HTML,
            )

    except RuntimeError as exc:
        logger.exception("ffmpeg error for user %d", user.id)
        await message.reply_text(
            f"❌ <b>Conversion error:</b>\n<code>{_e(str(exc)[:400])}</code>",
            parse_mode=ParseMode.HTML,
        )
        _cleanup(raw_path, mp3_path, trimmed_path)
    except Exception as exc:
        logger.exception("Audio pipeline error for user %d", user.id)
        await message.reply_text(
            f"❌ <code>{_e(type(exc).__name__)}: {_e(str(exc)[:300])}</code>",
            parse_mode=ParseMode.HTML,
        )
        _cleanup(raw_path, mp3_path, trimmed_path)


# ── Transcription helpers (same pattern as telegram-audio-bot) ────────────────

async def send_long_text(message, header: str, body: str) -> None:
    """Send text to Telegram, splitting into multiple messages if > 4000 chars.
    Header is HTML-formatted; body is sent with HTML-escaped content."""
    LIMIT = 4000
    full = header + html.escape(body)
    if len(full) <= LIMIT:
        await message.reply_text(full, parse_mode=ParseMode.HTML)
        return

    await message.reply_text(header, parse_mode=ParseMode.HTML)
    remaining = body
    part = 1
    while remaining:
        chunk = remaining[:LIMIT]
        remaining = remaining[LIMIT:]
        suffix = f"\n\n(part {part})" if remaining else ""
        await message.reply_text(chunk + suffix)
        part += 1


async def summarize_with_gemini(transcription: str, duration_mins: float) -> str:
    """Generate a real-estate walkthrough summary using Gemini (google-genai SDK)."""
    from google import genai as google_genai

    if not GOOGLE_API_KEY:
        raise ValueError("GOOGLE_API_KEY is not set")

    client = google_genai.Client(api_key=GOOGLE_API_KEY)

    prompt = (
        f"You are summarizing a {duration_mins:.1f}-minute real estate walkthrough recording "
        f"transcribed from mixed Farsi and English speech.\n\n"
        f"TRANSCRIPTION:\n{transcription}\n\n"
        f"Write a detailed summary of the property walkthrough. Follow these rules:\n"
        f"- Summarize in the same language(s) as the speaker — Farsi parts in Persian script (فارسی), English parts in English\n"
        f"- Use clear sections/bullet points to organize the key topics\n"
        f"- Cover all main points: property features, rooms, condition, highlights\n"
        f"- Keep it concise but comprehensive\n"
        f"- If mostly Farsi, write the summary mostly in Farsi; if mostly English, mostly English"
    )

    candidate_models = [
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash-001",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
    ]

    loop = asyncio.get_running_loop()
    last_error = None
    for model_name in candidate_models:
        try:
            response = await loop.run_in_executor(
                None,
                lambda m=model_name: client.models.generate_content(
                    model=m,
                    contents=prompt,
                )
            )
            logger.info("Summary generated using model: %s", model_name)
            return response.text.strip()
        except Exception as e:
            logger.warning("Model %s failed: %s", model_name, e)
            last_error = e
            continue

    raise RuntimeError(f"All summary models failed. Last error: {last_error}")


# ── Package delivery + full transcription + Gemini summary ────────────────────

async def _deliver_package(
    update: Update, context: ContextTypes.DEFAULT_TYPE
) -> None:
    message    = update.message
    user       = message.from_user
    video_info = context.user_data.pop(_VIDEO_KEY)
    audio_info = context.user_data.pop(_AUDIO_PROCESSED)

    safe_name      = audio_info["safe_name"]
    audio_filename = audio_info["audio_filename"]
    mp3_path       = audio_info["mp3_path"]
    video_filename = f"{safe_name}.mp4"

    # ── 1. Deliver video + renamed MP3 ────────────────────────────────────────
    try:
        await context.bot.send_chat_action(chat_id=message.chat_id, action=ChatAction.TYPING)

        caption = (
            f"📹 <b>{_e(video_filename)}</b>\n"
            f"<code>{_e(safe_name)}</code>"
        )
        if video_info["is_document"]:
            await context.bot.send_document(
                chat_id=message.chat_id,
                document=video_info["file_id"],
                caption=caption,
                parse_mode=ParseMode.HTML,
            )
        else:
            await context.bot.send_video(
                chat_id=message.chat_id,
                video=video_info["file_id"],
                caption=caption,
                parse_mode=ParseMode.HTML,
            )
        logger.info("Video re-sent via file_id: %s", video_filename)

        with open(mp3_path, "rb") as f:
            await context.bot.send_document(
                chat_id=message.chat_id,
                document=InputFile(f, filename=audio_filename),
                caption=f"🎵 <code>{_e(audio_filename)}</code>",
                parse_mode=ParseMode.HTML,
            )

        await message.reply_text(
            "✅ <b>Done!</b> Both files delivered.\n"
            "<i>Now transcribing the full recording...</i>",
            parse_mode=ParseMode.HTML,
        )

    except Exception as exc:
        logger.exception("Delivery error for user %d", user.id)
        await message.reply_text(
            f"❌ <b>Delivery error:</b> <code>{_e(type(exc).__name__)}: {_e(str(exc)[:300])}</code>",
            parse_mode=ParseMode.HTML,
        )
        _cleanup(mp3_path)
        return

    # ── 2. Full Chirp 3 transcription + Gemini summary ────────────────────────
    if os.getenv("GOOGLE_CLOUD_CREDENTIALS") and os.path.exists(mp3_path):
        duration_mins = get_audio_duration_secs(mp3_path) / 60

        trans_status = await message.reply_text(
            "🎙️ <b>Transcribing full audio with Chirp 3...</b>\n"
            "⏳ This takes 1–3 min for longer recordings",
            parse_mode=ParseMode.HTML,
        )
        try:
            transcription = await transcribe_audio(mp3_path)
            await trans_status.delete()

            header = f"📝 <b>Transcription</b> — ⏱ {duration_mins:.1f} min\n{'━' * 28}\n\n"
            await send_long_text(message, header, transcription)
            logger.info("Transcription sent for %s", safe_name)

            # ── Gemini summary ─────────────────────────────────────────────────
            if GOOGLE_API_KEY:
                sum_status = await message.reply_text(
                    "🤖 <b>Generating AI summary with Gemini...</b>",
                    parse_mode=ParseMode.HTML,
                )
                try:
                    summary = await summarize_with_gemini(transcription, duration_mins)
                    await sum_status.delete()
                    header = f"📋 <b>Summary</b> — ⏱ {duration_mins:.1f} min\n{'━' * 28}\n\n"
                    await send_long_text(message, header, summary)
                    logger.info("Summary sent for %s", safe_name)
                except Exception as sum_err:
                    await sum_status.edit_text(
                        f"⚠️ Summary failed: {_e(str(sum_err)[:200])}\n"
                        "✅ Transcription was still sent above."
                    )
                    logger.error("Summary error: %s", sum_err, exc_info=True)

        except Exception as trans_err:
            await trans_status.edit_text(
                f"❌ Transcription failed: {_e(str(trans_err)[:200])}\n"
                "✅ Files were still delivered above."
            )
            logger.error("Transcription error: %s", trans_err, exc_info=True)

    _cleanup(mp3_path)


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN not set — check your .env file.")

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start",  cmd_start))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("reset",  cmd_reset))

    app.add_handler(
        MessageHandler(
            filters.AUDIO | filters.VOICE | filters.Document.AUDIO,
            handle_audio,
        )
    )
    app.add_handler(
        MessageHandler(filters.VIDEO | filters.Document.VIDEO, handle_video)
    )

    logger.info("🏠 Real Estate File Renamer Bot running...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
