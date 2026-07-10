"""Presence tracking — derives the 'active at desk' boolean from recent activity.

Activity sources (all on the Pi 5): camera motion detection, chat messages,
and button presses. active = any activity within ACTIVE_WINDOW_SECONDS.
"""

import threading
import time as _time
from config import ACTIVE_WINDOW_SECONDS


class ActiveTracker:
    def __init__(self):
        self._lock = threading.Lock()
        self._last_activity = 0.0

    def touch(self):
        with self._lock:
            self._last_activity = _time.time()

    def is_active(self) -> bool:
        with self._lock:
            return (_time.time() - self._last_activity) < ACTIVE_WINDOW_SECONDS
