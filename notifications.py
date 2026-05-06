"""Notification store — proposed, approved, and recurring notifications with decay."""

import os
import json
import time as _time
from config import PROJECT_DIR, MAX_FIRINGS_PER_HOUR

NOTIFICATIONS_FILE = os.path.join(PROJECT_DIR, "notifications.json")

CATEGORIES = ["health", "productivity", "time", "environment", "misc"]


class NotificationStore:
    def __init__(self):
        self.notifications = []
        self.category_scores = {c: 0.0 for c in CATEGORIES}
        self._last_fire_time = 0.0
        self._load()

    def _load(self):
        if not os.path.exists(NOTIFICATIONS_FILE):
            return
        try:
            with open(NOTIFICATIONS_FILE, "r") as f:
                data = json.load(f)
            self.notifications = data.get("notifications", [])
            raw_scores = data.get("category_scores", {})
            self.category_scores = {c: float(raw_scores.get(c, 0.0)) for c in CATEGORIES}
            for n in self.notifications:
                if n.get("last_fired") and n["last_fired"] > self._last_fire_time:
                    self._last_fire_time = n["last_fired"]
            print(f"[NOTIF] Loaded {len(self.notifications)} notifications")
        except Exception as e:
            print(f"[NOTIF] Load error: {e}")

    def _save(self):
        try:
            with open(NOTIFICATIONS_FILE, "w") as f:
                json.dump(
                    {
                        "notifications": self.notifications,
                        "category_scores": self.category_scores,
                    },
                    f,
                )
        except Exception as e:
            print(f"[NOTIF] Save error: {e}")

    def create_proposal(self, message, category, trigger_type, trigger_value):
        notif = {
            "id": f"notif_{int(_time.time())}",
            "status": "proposed",
            "message": message,
            "category": category,
            "trigger_type": trigger_type,
            "trigger_value": trigger_value,
            "proposed_at": _time.time(),
            "decided_at": None,
            "last_fired": None,
            "fire_count": 0,
            "decay_score": 1.0,
        }
        self.notifications.append(notif)
        self._save()
        return notif

    def approve_pending(self):
        for n in self.notifications:
            if n["status"] == "proposed":
                n["status"] = "approved"
                n["decided_at"] = _time.time()
                cat = n["category"]
                self.category_scores[cat] = min(1.0, self.category_scores.get(cat, 0.0) + 0.2)
                self._save()
                return n
        return None

    def reject_pending(self):
        for n in self.notifications:
            if n["status"] == "proposed":
                n["status"] = "rejected"
                n["decided_at"] = _time.time()
                cat = n["category"]
                self.category_scores[cat] = max(-1.0, self.category_scores.get(cat, 0.0) - 0.3)
                self._save()
                return n
        return None

    def expire_pending(self):
        for n in self.notifications:
            if n["status"] == "proposed":
                n["status"] = "rejected"
                n["decided_at"] = _time.time()
                cat = n["category"]
                self.category_scores[cat] = max(-1.0, self.category_scores.get(cat, 0.0) - 0.1)
                self._save()
                return n
        return None

    def get_due_notification(self):
        now = _time.time()
        if now - self._last_fire_time < (3600 // MAX_FIRINGS_PER_HOUR):
            return None

        due = []
        for n in self.notifications:
            if n["status"] != "approved":
                continue

            if n["trigger_type"] == "interval":
                interval = int(n["trigger_value"])
                if n["last_fired"] is None or now - n["last_fired"] >= interval:
                    due.append(n)
            elif n["trigger_type"] == "time_of_day":
                current_time = _time.strftime("%H:%M")
                if n["trigger_value"] == current_time:
                    if n["last_fired"] is None:
                        due.append(n)
                    else:
                        last_day = _time.strftime("%Y-%m-%d", _time.localtime(n["last_fired"]))
                        today = _time.strftime("%Y-%m-%d")
                        if last_day != today:
                            due.append(n)

        if not due:
            return None

        due.sort(key=lambda n: n["decay_score"], reverse=True)
        return due[0]

    def record_firing(self, notification_id):
        for n in self.notifications:
            if n["id"] == notification_id:
                n["last_fired"] = _time.time()
                n["fire_count"] += 1
                self._last_fire_time = n["last_fired"]
                self._save()
                return

    def record_acknowledgment(self, notification_id):
        for n in self.notifications:
            if n["id"] == notification_id:
                n["decay_score"] = min(1.0, n["decay_score"] + 0.1)
                self._save()
                return

    def decay_unacknowledged(self, notification_id):
        for n in self.notifications:
            if n["id"] == notification_id:
                n["decay_score"] -= 0.3
                if n["decay_score"] < 0.2:
                    n["status"] = "expired"
                self._save()
                return

    def has_pending_proposal(self):
        return any(n["status"] == "proposed" for n in self.notifications)

    def get_review_summary(self, patterns=None, cooldown_categories=None):
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
                unit = "min" if n["trigger_type"] == "interval" else ""
                lines.append(
                    f'"{n["message"]}" ({n["category"]}, every {n["trigger_value"]}{unit}, last fired {last_str})'
                )
            parts.append(f"Active notifications ({len(active)}): {', '.join(lines)}.")

        expired = [n for n in self.notifications if n["status"] == "expired"]
        if expired:
            msgs = ", ".join('"' + n["message"] + '"' for n in expired)
            parts.append(
                f"Expired notifications ({len(expired)}): {msgs}."
            )

        scores = ", ".join(
            f"{c}={self.category_scores.get(c, 0.0):.1f}" for c in CATEGORIES
        )
        parts.append(f"Category scores: {scores}.")

        if cooldown_categories:
            cd = ", ".join(
                f"{c} ({v} reviews)" for c, v in cooldown_categories.items()
            )
            parts.append(f"Cooldowns: {cd}.")

        if patterns:
            parts.append(f"Patterns detected: {patterns}.")

        parts.append(
            "-> If you see a pattern worth a notification, call propose_notification. Otherwise continue your rhythm."
        )

        return "\n".join(parts)
