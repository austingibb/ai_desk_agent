"""Piper TTS — non-blocking speech via piper HTTP server. Fire-and-forget like sounds.py.

Requires piper HTTP server running: python3 -m piper.http_server -m <voice>
"""

import io
import os
import signal
import subprocess
import threading

import requests

from config import ENABLE_TTS, PIPER_HTTP_URL

_current_process = None
_lock = threading.Lock()


def speak(text: str):
    """Speak text via Piper. Non-blocking: spawns a daemon thread.
    Interrupts any in-flight speech first."""
    if not ENABLE_TTS or not text or not text.strip():
        return
    interrupt()
    t = threading.Thread(target=_run_speech, args=(text,), daemon=True)
    t.start()


def interrupt():
    """Kill any in-flight playback (ffplay/aplay)."""
    global _current_process
    with _lock:
        proc = _current_process
        _current_process = None
    if proc and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass


def _sanitize(text: str) -> str:
    """Strip content that shouldn't be spoken."""
    text = text.replace("\n", " ").replace("\r", " ")
    text = " ".join(text.split())
    for suffix in ["(full message on chat)"]:
        text = text.replace(suffix, "")
    return text.strip()


def _run_speech(text: str):
    """Fetch WAV from piper HTTP server and play it."""
    global _current_process
    text = _sanitize(text)
    if not text:
        return
    proc = None
    try:
        resp = requests.post(
            PIPER_HTTP_URL,
            json={"text": text},
            timeout=30,
        )
        if resp.status_code != 200:
            return
        proc = subprocess.Popen(
            ["aplay", "-q"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )
        with _lock:
            _current_process = proc
        proc.communicate(input=resp.content)
    except Exception:
        pass
    finally:
        with _lock:
            if _current_process is proc:
                _current_process = None
