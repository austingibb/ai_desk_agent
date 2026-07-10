"""Caffeine drink store — append-only log of drinks, pruned to the last 24h."""

import os
import json
import threading
import time as _time
from config import PROJECT_DIR, DRINK_RETENTION_SECONDS
from logger import info

DRINKS_FILE = os.path.join(PROJECT_DIR, "drinks.json")


class DrinkStore:
    def __init__(self):
        self.drinks = []  # [{"t": epoch_ms, "mg": int, "label": str}]
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        if not os.path.exists(DRINKS_FILE):
            return
        try:
            with open(DRINKS_FILE, "r") as f:
                data = json.load(f)
            self.drinks = data.get("drinks", [])
            info(f"[CAFFEINE] Loaded {len(self.drinks)} drinks")
            self._prune()
        except Exception as e:
            info(f"[CAFFEINE] Load error: {e}")

    def _save(self):
        try:
            with open(DRINKS_FILE, "w") as f:
                json.dump({"drinks": self.drinks}, f)
        except Exception as e:
            info(f"[CAFFEINE] Save error: {e}")

    def _prune(self):
        """Drop drinks older than the retention window. Caller must hold no lock."""
        cutoff_ms = int((_time.time() - DRINK_RETENTION_SECONDS) * 1000)
        before = len(self.drinks)
        self.drinks = [d for d in self.drinks if d.get("t", 0) >= cutoff_ms]
        removed = before - len(self.drinks)
        if removed:
            info(f"[CAFFEINE] Pruned {removed} drinks older than 24h")
            self._save()

    def add(self, mg: int, label: str, minutes_ago: int = 0) -> dict:
        """Append a drink. minutes_ago backdates it; timestamps never land in the future."""
        minutes_ago = max(0, int(minutes_ago))
        t = int((_time.time() - minutes_ago * 60) * 1000)
        entry = {"t": t, "mg": int(mg), "label": label}
        with self._lock:
            self.drinks.append(entry)
            self.drinks.sort(key=lambda d: d.get("t", 0))
            self._save()
        return entry

    def get_feed_drinks(self) -> list:
        """Drinks for the public feed: last 24h only, t/mg fields only, ascending."""
        now_ms = int(_time.time() * 1000)
        with self._lock:
            self._prune()
            return [
                {"t": min(d["t"], now_ms), "mg": d["mg"]}
                for d in self.drinks
            ]

    def total_recent_mg(self) -> int:
        with self._lock:
            self._prune()
            return sum(d["mg"] for d in self.drinks)
