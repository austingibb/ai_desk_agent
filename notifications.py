"""Notification store — proposed, approved, and recurring notifications."""

import os
import json
import time as _time
from config import PROJECT_DIR
from logger import info

NOTIFICATIONS_FILE = os.path.join(PROJECT_DIR, "notifications.json")

# Dead notifications older than 24h are purged on load
PURGE_AGE_SECONDS = 86400


class NotificationStore:
    def __init__(self):
        self.notifications = []
        self._last_fire_time = 0.0
        self._load()

    def _load(self):
        if not os.path.exists(NOTIFICATIONS_FILE):
            return
        try:
            with open(NOTIFICATIONS_FILE, "r") as f:
                data = json.load(f)
            self.notifications = data.get("notifications", [])
            # Migrate: drop legacy fields
            for n in self.notifications:
                n.pop("decay_score", None)
                n.pop("category", None)
            for n in self.notifications:
                if n.get("last_fired") and n["last_fired"] > self._last_fire_time:
                    self._last_fire_time = n["last_fired"]
            info(f"[NOTIF] Loaded {len(self.notifications)} notifications")
            self._purge_dead()
        except Exception as e:
            info(f"[NOTIF] Load error: {e}")

    def _purge_dead(self):
        """Remove rejected/expired notifications older than PURGE_AGE_SECONDS."""
        now = _time.time()
        before = len(self.notifications)
        self.notifications = [
            n for n in self.notifications
            if n["status"] not in ("rejected", "expired")
            or (now - (n.get("decided_at") or n.get("proposed_at") or 0)) < PURGE_AGE_SECONDS
        ]
        removed = before - len(self.notifications)
        if removed:
            info(f"[NOTIF] Purged {removed} dead notifications")
            self._save()

    def _save(self):
        try:
            with open(NOTIFICATIONS_FILE, "w") as f:
                json.dump({"notifications": self.notifications}, f)
        except Exception as e:
            info(f"[NOTIF] Save error: {e}")

    def create_proposal(self, message, trigger_type, trigger_value):
        notif = {
            "id": f"notif_{int(_time.time())}",
            "status": "proposed",
            "message": message,
            "trigger_type": trigger_type,
            "trigger_value": trigger_value,
            "proposed_at": _time.time(),
            "decided_at": None,
            "last_fired": None,
            "fire_count": 0,
        }
        self.notifications.append(notif)
        self._save()
        return notif

    def approve_pending(self):
        for n in self.notifications:
            if n["status"] == "proposed":
                n["status"] = "approved"
                n["decided_at"] = _time.time()
                n["next_fire"] = _time.time()
                self._save()
                return n
        return None

    def reject_pending(self):
        for n in self.notifications:
            if n["status"] == "proposed":
                n["status"] = "rejected"
                n["decided_at"] = _time.time()
                self._save()
                return n
        return None

    def get_due_notification(self):
        now = _time.time()
        if now - self._last_fire_time < 300:
            return None

        due = []
        for n in self.notifications:
            if n["status"] != "approved":
                continue
            next_fire = n.get("next_fire")
            if next_fire is not None and now >= next_fire:
                due.append(n)

        if not due:
            return None

        due.sort(key=lambda n: n.get("next_fire", 0))
        return due[0]

    def record_firing(self, notification_id):
        for n in self.notifications:
            if n["id"] == notification_id:
                n["last_fired"] = _time.time()
                n["fire_count"] += 1
                n["next_fire"] = None
                self._last_fire_time = n["last_fired"]
                self._save()
                return

    def schedule(self, notification_id, seconds):
        for n in self.notifications:
            if n["id"] == notification_id:
                n["next_fire"] = _time.time() + seconds
                self._save()
                return True
        return False

    def delete(self, notification_id):
        self.notifications = [n for n in self.notifications if n["id"] != notification_id]
        self._save()

    def has_pending_proposal(self):
        return any(n["status"] == "proposed" for n in self.notifications)

    def get_review_summary(self, patterns=None):
        now = _time.strftime("%-I:%M%p").lower().lstrip("0")
        day = _time.strftime("%a")
        parts = [f"[Notification review] Time: {now} {day}."]

        active = [n for n in self.notifications if n["status"] == "approved"]
        if active:
            lines = []
            for n in active:
                last = n.get("last_fired")
                last_str = (
                    _time.strftime("%-I:%M%p", _time.localtime(last)).lower().lstrip("0")
                    if last
                    else "never"
                )
                next_fire = n.get("next_fire")
                if next_fire is None:
                    schedule_str = "UNSCHEDULED — call schedule_notification to set next fire time"
                else:
                    mins_left = max(0, int((next_fire - _time.time()) / 60))
                    schedule_str = f"next fire in ~{mins_left}min"
                lines.append(
                    f'id={n["id"]} "{n["message"]}" (last fired {last_str}, {schedule_str})'
                )
            parts.append(f"Active notifications ({len(active)}):\n" + "\n".join(f"  - {l}" for l in lines))

        if patterns:
            parts.append(f"Patterns detected: {patterns}.")

        parts.append(
            "-> You can propose_notification, schedule_notification, or delete_notification. Or do nothing."
        )

        return "\n".join(parts)
