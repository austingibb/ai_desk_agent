"""Non-blocking sound playback for tool events."""

import os
import subprocess

SOUNDS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sounds")

# Map event names to (filename, volume 0.0-1.0)
EVENT_SOUNDS = {
    "update_display": ("display.wav", 1.0),
    "take_photo": ("shutter.wav", 1.0),
    "search": ("search.wav", 0.5),
    "wait": ("wait.wav", 0.6),
    "thinking": ("thinking.wav", 1.0),
}


def play(event: str):
    """Play the sound for an event. Non-blocking, fails silently."""
    entry = EVENT_SOUNDS.get(event)
    if not entry:
        return
    filename, volume = entry
    path = os.path.join(SOUNDS_DIR, filename)
    if not os.path.exists(path):
        return
    try:
        vol = str(int(volume * 65536))
        subprocess.Popen(
            ["paplay", f"--volume={vol}", path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass
