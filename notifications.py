"""Notification store — proposed, approved, and recurring notifications with decay."""

import os
import json
import time as _time
from config import PROJECT_DIR

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
                n["next_fire"] = _time.time()  # fire immediately on first approval
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
        # Global cooldown: don't fire any notification within 5 min of the last one
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
                n["next_fire"] = None  # AI must schedule the next one
                self._last_fire_time = n["last_fired"]
                self._save()
                return

    def schedule(self, notification_id, seconds):
        for n in self.notifications:
            if n["id"] == notification_id:
                n["next_fire"] = _time.time() + seconds
                self._save()
                return n
        return None

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
                next_fire = n.get("next_fire")
                if next_fire is None:
                    schedule_str = "UNSCHEDULED — call schedule_notification to set next fire time"
                else:
                    mins_left = max(0, int((next_fire - _time.time()) / 60))
                    schedule_str = f"next fire in ~{mins_left}min"
                lines.append(
                    f'id={n["id"]} "{n["message"]}" ({n["category"]}, last fired {last_str}, {schedule_str})'
                )
            parts.append(f"Active notifications ({len(active)}):\n" + "\n".join(f"  - {l}" for l in lines))

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
