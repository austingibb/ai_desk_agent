"""Caffeine drink store — append-only log of drinks, pruned to the retention window (30 days)."""

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

    def _read_file(self) -> list:
        """Current on-disk drinks; [] if the file is missing or unreadable."""
        if not os.path.exists(DRINKS_FILE):
            return []
        try:
            with open(DRINKS_FILE, "r") as f:
                return json.load(f).get("drinks", [])
        except Exception as e:
            info(f"[CAFFEINE] Read error: {e}")
            return []

    def _load(self):
        self.drinks = self._read_file()
        if self.drinks:
            info(f"[CAFFEINE] Loaded {len(self.drinks)} drinks")
        self._prune()

    def _save(self):
        """Re-read the file and merge before writing, so entries added to
        drinks.json outside this process are never clobbered. Entries older
        than the retention window are not merged back."""
        cutoff_ms = int((_time.time() - DRINK_RETENTION_SECONDS) * 1000)
        seen = {(d.get("t"), d.get("mg"), d.get("label")) for d in self.drinks}
        external = [
            d for d in self._read_file()
            if d.get("t", 0) >= cutoff_ms
            and (d.get("t"), d.get("mg"), d.get("label")) not in seen
        ]
        if external:
            info(f"[CAFFEINE] Merged {len(external)} external entries from disk")
            self.drinks.extend(external)
            self.drinks.sort(key=lambda d: d.get("t", 0))
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
            info(f"[CAFFEINE] Pruned {removed} drinks past retention")
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
        """Drinks for the public feed: retention window only, t/mg fields only, ascending."""
        now_ms = int(_time.time() * 1000)
        with self._lock:
            self._prune()
            return [
                {"t": min(d["t"], now_ms), "mg": d["mg"]}
                for d in self.drinks
            ]

    def total_recent_mg(self) -> int:
        """Sum of all drinks within the retention window (up to 30 days), not just 24h."""
        with self._lock:
            self._prune()
            return sum(d["mg"] for d in self.drinks)

    def total_last_24h_mg(self) -> int:
        """Sum of drinks in the last 24 actual hours."""
        cutoff_ms = int((_time.time() - 86400) * 1000)
        with self._lock:
            self._prune()
            return sum(d["mg"] for d in self.drinks if d.get("t", 0) >= cutoff_ms)

    def list_recent(self, n: int = 20) -> list:
        with self._lock:
            self._prune()
            return list(self.drinks[-n:])

    def edit(self, timestamp_ms: int, mg: int | None = None, label: str | None = None) -> dict | None:
        with self._lock:
            for d in self.drinks:
                if d.get("t") == timestamp_ms:
                    if mg is not None:
                        d["mg"] = int(mg)
                    if label is not None:
                        d["label"] = label
                    self._save()
                    return dict(d)
        return None
