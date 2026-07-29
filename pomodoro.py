"""Pomodoro store — append-only log of completed pomodoro cycles, with stats.

Mirrors caffeine.py's DrinkStore: thread-safe, merges external writes on save,
editable after the fact. Unlike drinks it is NOT pruned to a retention window —
best-streak and averages need the full history, and the log stays tiny.

Each entry: {"t": epoch_ms, "label": str}. A "cycle" is one completed pomodoro
(25 min work + 5 min break). Daily/weekly counts and streaks are computed in the
machine's LOCAL timezone, so "today" rolls over at local midnight with no cron.
"""

import os
import json
import threading
import time as _time
from datetime import date, datetime, timedelta
from config import PROJECT_DIR
from logger import info

POMODOROS_FILE = os.path.join(PROJECT_DIR, "pomodoros.json")


def _local_date(t_ms: int) -> date:
    return datetime.fromtimestamp(t_ms / 1000).date()


class PomodoroStore:
    def __init__(self):
        self.cycles = []  # [{"t": epoch_ms, "label": str}]
        self._lock = threading.Lock()
        self._load()

    def _read_file(self) -> list:
        """Current on-disk cycles; [] if the file is missing or unreadable."""
        if not os.path.exists(POMODOROS_FILE):
            return []
        try:
            with open(POMODOROS_FILE, "r") as f:
                return json.load(f).get("cycles", [])
        except Exception as e:
            info(f"[POMODORO] Read error: {e}")
            return []

    def _load(self):
        self.cycles = self._read_file()
        self.cycles.sort(key=lambda c: c.get("t", 0))
        if self.cycles:
            info(f"[POMODORO] Loaded {len(self.cycles)} cycles")

    def _save(self):
        """Write the in-memory log authoritatively. Unlike DrinkStore we do NOT
        merge entries back from disk: this store is single-writer (the
        orchestrator), and merging would resurrect entries removed by delete()."""
        try:
            with open(POMODOROS_FILE, "w") as f:
                json.dump({"cycles": self.cycles}, f)
        except Exception as e:
            info(f"[POMODORO] Save error: {e}")

    def add(self, label: str = "pomodoro", minutes_ago: int = 0) -> dict:
        """Append a completed cycle. minutes_ago backdates it; never lands in the future."""
        minutes_ago = max(0, int(minutes_ago))
        t = int((_time.time() - minutes_ago * 60) * 1000)
        entry = {"t": t, "label": label or "pomodoro"}
        with self._lock:
            self.cycles.append(entry)
            self.cycles.sort(key=lambda c: c.get("t", 0))
            self._save()
        return entry

    def edit(self, timestamp_ms: int, label: str | None = None,
             new_timestamp_ms: int | None = None) -> dict | None:
        with self._lock:
            for c in self.cycles:
                if c.get("t") == timestamp_ms:
                    if label is not None:
                        c["label"] = label
                    if new_timestamp_ms is not None:
                        c["t"] = int(new_timestamp_ms)
                    self.cycles.sort(key=lambda c: c.get("t", 0))
                    self._save()
                    return dict(c)
        return None

    def delete(self, timestamp_ms: int) -> bool:
        with self._lock:
            before = len(self.cycles)
            self.cycles = [c for c in self.cycles if c.get("t") != timestamp_ms]
            if len(self.cycles) < before:
                self._save()
                return True
        return False

    def list_recent(self, n: int = 20) -> list:
        with self._lock:
            return list(self.cycles[-n:])

    def _active_dates(self) -> set:
        return {_local_date(c["t"]) for c in self.cycles if c.get("t")}

    def stats(self) -> dict:
        """Snapshot of the tracked metrics, computed in local time."""
        with self._lock:
            cycles = list(self.cycles)
        if not cycles:
            return {
                "total": 0, "today": 0, "this_week": 0,
                "current_streak": 0, "best_streak": 0, "avg_per_day": 0.0,
            }

        today = date.today()
        active = {_local_date(c["t"]) for c in cycles}

        today_count = sum(1 for c in cycles if _local_date(c["t"]) == today)
        wk = today.isocalendar()[:2]
        week_count = sum(1 for c in cycles if _local_date(c["t"]).isocalendar()[:2] == wk)

        # Current streak: count back from today, or from yesterday if today has
        # nothing yet (the streak is still alive until a full day is missed).
        if today in active:
            cursor = today
        elif (today - timedelta(days=1)) in active:
            cursor = today - timedelta(days=1)
        else:
            cursor = None
        current_streak = 0
        while cursor is not None and cursor in active:
            current_streak += 1
            cursor -= timedelta(days=1)

        # Best streak: longest run of consecutive active days ever.
        best_streak = 0
        run = 0
        prev = None
        for d in sorted(active):
            if prev is not None and d - prev == timedelta(days=1):
                run += 1
            else:
                run = 1
            best_streak = max(best_streak, run)
            prev = d

        first_day = min(active)
        span_days = (today - first_day).days + 1
        avg_per_day = round(len(cycles) / span_days, 2) if span_days > 0 else 0.0

        return {
            "total": len(cycles),
            "today": today_count,
            "this_week": week_count,
            "current_streak": current_streak,
            "best_streak": best_streak,
            "avg_per_day": avg_per_day,
        }

    def count_today(self) -> int:
        today = date.today()
        with self._lock:
            return sum(1 for c in self.cycles if _local_date(c["t"]) == today)
