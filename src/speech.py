import os
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
import threading
from io import BytesIO
from gtts import gTTS

try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False

_mixer_initialized = False
_play_lock = threading.Lock()

def is_bengali_text(text):
    """Returns True if text contains Bengali characters."""
    return any("\u0980" <= c <= "\u09ff" for c in str(text))

def _init_mixer():
    global _mixer_initialized
    if not HAS_PYGAME:
        return False
    if not _mixer_initialized:
        try:
            pygame.mixer.init()
            _mixer_initialized = True
        except Exception as e:
            print("Pygame mixer init notice:", e)
            return False
    return _mixer_initialized

def make_audio(text, lang="bn"):
    if not text or not str(text).strip():
        return None
    try:
        # Auto-switch language based on text characters if needed
        clean_text = str(text).strip()
        actual_lang = "bn" if is_bengali_text(clean_text) else ("en" if lang == "en" else lang)
        
        f = BytesIO()
        gTTS(text=clean_text, lang=actual_lang).write_to_fp(f)
        f.seek(0)
        return f.getvalue()
    except Exception as e:
        print("gTTS error:", e)
        # Fallback to english if language failed
        try:
            f = BytesIO()
            gTTS(text=str(text).strip(), lang="en").write_to_fp(f)
            f.seek(0)
            return f.getvalue()
        except Exception:
            return None

def is_speaking():
    if not HAS_PYGAME:
        return False
    if not _init_mixer():
        return False
    try:
        return pygame.mixer.music.get_busy()
    except Exception:
        return False

def play_audio_bytes(audio_bytes):
    if not audio_bytes:
        return
    if not _init_mixer():
        return
    with _play_lock:
        try:
            bio = BytesIO(audio_bytes)
            pygame.mixer.music.load(bio)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
        except Exception as e:
            print("Audio playback error:", e)

def speak(text, lang=None, async_mode=True):
    if not text or not str(text).strip():
        return
    if is_speaking():
        return  # Skip if currently speaking to avoid overlapping

    clean_text = str(text).strip()
    # If lang is not specified, auto-detect
    if lang is None:
        lang = "bn" if is_bengali_text(clean_text) else "en"

    def _worker():
        try:
            audio_bytes = make_audio(clean_text, lang=lang)
            if audio_bytes:
                play_audio_bytes(audio_bytes)
        except Exception as e:
            print("Speak worker error:", e)

    if async_mode:
        t = threading.Thread(target=_worker, daemon=True)
        t.start()
    else:
        _worker()
