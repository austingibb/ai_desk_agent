"""Non-blocking sound playback for tool events."""

import os
import subprocess

SOUNDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sounds")

# Map event names to filenames in sounds/
EVENT_SOUNDS = {
    "update_display": "display.wav",
    "take_photo": "shutter.wav",
    "search": "search.wav",
    "wait": "wait.wav",
}


def play(event: str):
    """Play the sound for an event. Non-blocking, fails silently."""
    filename = EVENT_SOUNDS.get(event)
    if not filename:
        return
    path = os.path.join(SOUNDS_DIR, filename)
    if not os.path.exists(path):
        return
    try:
        subprocess.Popen(
            ["aplay", "-q", path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass
