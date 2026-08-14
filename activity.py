"""Persistent, camera-derived AK/AFK activity timeline."""

import copy
import json
import os
import re
import tempfile
import threading
import time as _time

from config import (
    ACTIVITY_CONFIRMATION_OBSERVATIONS,
    ACTIVITY_LOG_FILE,
    ACTIVITY_RETENTION_SECONDS,
)
from logger import info


ACTIVITY_PRESENCE = {
    "working_at_computer": "AK",
    "out": "AFK",
    "relaxing": "AFK",
    "sleeping": "AFK",
    "eating": "AFK",
    "exercising": "AFK",
    "chores": "AFK",
}
ACTIVITIES = tuple(ACTIVITY_PRESENCE)

_EMPTY_PATTERNS = (
    r"\broom (?:is |appears )?empty\b",
    r"\bno (?:one|people|person) (?:is |are )?(?:present|visible|in (?:the )?room)\b",
    r"\bnobody (?:is )?(?:present|visible|in (?:the )?room)\b",
    r"\bunoccupied\b",
)
_PERSON_PATTERN = re.compile(
    r"\b(person|people|man|woman|someone|individual|occupant|user)\b",
    re.IGNORECASE,
)
_ACTIVITY_PATTERNS = {
    # Order is intentional. Eating wins even when it happens at the keyboard.
    "eating": (
        r"\b(eating|having (?:a )?(?:meal|snack|breakfast|lunch|dinner)|"
        r"taking bites?|consuming food|snacking)\b",
    ),
    "exercising": (
        r"\b(exercising|working out|doing (?:yoga|push[ -]?ups|sit[ -]?ups|"
        r"squats?|stretches)|lifting weights?|on (?:a )?treadmill)\b",
    ),
    "chores": (
        r"\b(cleaning|vacuuming|sweeping|mopping|doing laundry|folding (?:clothes|"
        r"laundry)|washing dishes|making (?:the )?bed|tidying|organizing the room|"
        r"housework|chores?)\b",
    ),
    "sleeping": (
        r"\b(sleeping|asleep|napping|taking (?:a )?nap)\b",
        r"\b(?:lying|laying) (?:in|on) (?:the )?bed\b.*\beyes closed\b",
    ),
    "working_at_computer": (
        r"\b(typing|using|working (?:at|on)|focused on|looking at|facing)\b.{0,45}"
        r"\b(keyboard|computer|laptop|monitor|screen)\b",
        r"\b(at (?:the|a) keyboard|using (?:the|a) mouse|computer work|"
        r"working at (?:the|a) computer)\b",
        r"\b(?:seated|sitting) at (?:the|a) desk\b.{0,45}"
        r"\b(computer|laptop|monitor|screen|keyboard)\b",
    ),
    "relaxing": (
        r"\b(relaxing|resting|lounging|watching (?:tv|television|a movie|a show)|"
        r"reading|sitting on (?:the|a) (?:couch|sofa)|lying (?:down|on (?:the|a) couch))\b",
    ),
}


def _matches(text: str, patterns) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def _json_object(text: str) -> dict | None:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines.pop()
        stripped = "\n".join(lines).strip()
    candidates = [stripped]
    candidates.extend(stripped[index:] for index, char in enumerate(stripped) if char == "{")
    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            value, _ = decoder.raw_decode(candidate.lstrip())
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(value, dict):
            return value
    return None


def _classification(activity: str, evidence: str, confidence: str = "high") -> dict:
    return {
        "presence": ACTIVITY_PRESENCE[activity],
        "activity": activity,
        "confidence": confidence,
        "evidence": " ".join(evidence.split())[:300],
    }


def _specific_activity(text: str) -> str | None:
    for activity, patterns in _ACTIVITY_PATTERNS.items():
        if _matches(text, patterns):
            return activity
    return None


def classify_scene(description: str) -> dict | None:
    """Classify a vision description into the bounded AK/AFK state model.

    Structured AARG responses are preferred. Generic prose is intentionally
    conservative: an ambiguous frame returns ``None`` so the current state is
    retained instead of inventing a transition.
    """
    if not isinstance(description, str) or not description.strip():
        return None

    structured = _json_object(description)
    if structured:
        people = structured.get("people_present", {})
        activity = structured.get("activity", {})
        people_status = str(
            people.get("status", "") if isinstance(people, dict) else people
        ).strip().lower()
        if people_status in {"no", "none", "absent", "false", "empty"}:
            evidence = people.get("evidence", "") if isinstance(people, dict) else ""
            return _classification("out", evidence or "No person visible in the room.")

        if isinstance(activity, dict):
            category = str(activity.get("category", ""))
            detail = str(activity.get("description", ""))
        else:
            category = str(activity)
            detail = ""
        evidence = " ".join(part for part in (category, detail) if part).strip()
        specific = _specific_activity(evidence.replace("_", " "))
        if specific:
            return _classification(specific, evidence)

        category_key = category.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "computer": "working_at_computer",
            "computer_use": "working_at_computer",
            "desk_work": "working_at_computer",
            "working_at_computer": "working_at_computer",
            "eating": "eating",
            "exercise": "exercising",
            "exercising": "exercising",
            "chores": "chores",
            "housework": "chores",
            "sleeping": "sleeping",
            "relaxing": "relaxing",
            "resting": "relaxing",
        }
        if category_key in aliases:
            return _classification(aliases[category_key], evidence or category_key)
        if people_status in {"yes", "present", "true", "one", "multiple"}:
            return _classification(
                "relaxing",
                evidence or "Person visible; specific activity unclear.",
                confidence="low",
            )

    text = " ".join(description.split())
    if _matches(text, _EMPTY_PATTERNS):
        return _classification("out", text)
    specific = _specific_activity(text)
    if specific:
        return _classification(specific, text)
    if _PERSON_PATTERN.search(text):
        return _classification(
            "relaxing", text, confidence="low"
        )
    return None


class ActivityStore:
    """Thread-safe activity segment store with persisted transition debounce."""

    VERSION = 1

    def __init__(
        self,
        file_path: str = ACTIVITY_LOG_FILE,
        retention_seconds: int = ACTIVITY_RETENTION_SECONDS,
        confirmation_observations: int = ACTIVITY_CONFIRMATION_OBSERVATIONS,
    ):
        self.file_path = file_path
        self.retention_seconds = max(1, int(retention_seconds))
        self.confirmation_observations = max(1, int(confirmation_observations))
        self.segments = []
        self.pending = None
        self._lock = threading.RLock()
        self._load()

    @staticmethod
    def _state_key(value: dict | None) -> tuple:
        if not isinstance(value, dict):
            return (None, None)
        return value.get("presence"), value.get("activity")

    def _read_file(self) -> dict:
        if not os.path.exists(self.file_path):
            return {}
        try:
            with open(self.file_path, "r", encoding="utf-8") as handle:
                value = json.load(handle)
            return value if isinstance(value, dict) else {}
        except Exception as exc:
            info(f"[ACTIVITY] Load error: {exc}")
            return {}

    def _load(self):
        data = self._read_file()
        segments = data.get("segments", [])
        if isinstance(segments, list):
            self.segments = [
                segment for segment in segments
                if isinstance(segment, dict)
                and segment.get("activity") in ACTIVITY_PRESENCE
                and segment.get("presence") == ACTIVITY_PRESENCE[segment["activity"]]
            ]
            self.segments.sort(key=lambda item: item.get("started_at", 0))
        pending = data.get("pending")
        if (
            isinstance(pending, dict)
            and pending.get("activity") in ACTIVITY_PRESENCE
            and pending.get("presence") == ACTIVITY_PRESENCE[pending["activity"]]
        ):
            self.pending = pending
        self._prune(_time.time())
        if self.segments:
            info(f"[ACTIVITY] Loaded {len(self.segments)} activity segment(s)")

    def _atomic_save(self):
        directory = os.path.dirname(os.path.abspath(self.file_path))
        os.makedirs(directory, exist_ok=True)
        fd, temp_path = tempfile.mkstemp(prefix=".activity-", dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "version": self.VERSION,
                        "segments": self.segments,
                        "pending": self.pending,
                    },
                    handle,
                    separators=(",", ":"),
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.file_path)
        except Exception:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise

    def _save(self):
        try:
            self._atomic_save()
        except Exception as exc:
            info(f"[ACTIVITY] Save error: {exc}")

    def _prune(self, now: float):
        cutoff = float(now) - self.retention_seconds
        self.segments = [
            segment for segment in self.segments
            if segment.get("ended_at") is None
            or float(segment.get("ended_at", 0)) >= cutoff
        ]

    def current(self) -> dict | None:
        with self._lock:
            if not self.segments or self.segments[-1].get("ended_at") is not None:
                return None
            return copy.deepcopy(self.segments[-1])

    def _start_segment(self, observation: dict, observed_at: float, source: str):
        current = self.current()
        if current:
            self.segments[-1]["ended_at"] = observed_at
        self.segments.append({
            "presence": observation["presence"],
            "activity": observation["activity"],
            "started_at": observed_at,
            "ended_at": None,
            "last_observed_at": observed_at,
            "source": source,
            "evidence": observation.get("evidence", ""),
        })
        self.pending = None

    def observe(
        self,
        observation: dict | None,
        *,
        observed_at: float | None = None,
        source: str = "main_camera",
    ) -> dict:
        """Apply a classified observation and return current state + change flag."""
        if not observation or observation.get("activity") not in ACTIVITY_PRESENCE:
            return {"changed": False, "current": self.current(), "ignored": True}
        activity = observation["activity"]
        expected_presence = ACTIVITY_PRESENCE[activity]
        if observation.get("presence") != expected_presence:
            return {"changed": False, "current": self.current(), "ignored": True}

        when = float(observed_at if observed_at is not None else _time.time())
        with self._lock:
            current = self.current()
            if current:
                when = max(when, float(current.get("last_observed_at", when)))
            if not current:
                self._start_segment(observation, when, source)
                changed = True
            elif self._state_key(current) == self._state_key(observation):
                segment = self.segments[-1]
                segment["last_observed_at"] = when
                segment["source"] = source
                segment["evidence"] = observation.get("evidence", "")
                self.pending = None
                changed = False
            else:
                immediate = observation.get("confidence") == "high"
                if self._state_key(self.pending) == self._state_key(observation):
                    self.pending["observations"] = int(
                        self.pending.get("observations", 1)
                    ) + 1
                    self.pending["last_observed_at"] = when
                    self.pending["source"] = source
                    self.pending["evidence"] = observation.get("evidence", "")
                else:
                    self.pending = {
                        "presence": observation["presence"],
                        "activity": activity,
                        "first_observed_at": when,
                        "last_observed_at": when,
                        "observations": 1,
                        "source": source,
                        "evidence": observation.get("evidence", ""),
                    }
                if immediate or self.pending["observations"] >= self.confirmation_observations:
                    self._start_segment(observation, when, source)
                    changed = True
                else:
                    changed = False
            self._prune(when)
            self._save()
            return {
                "changed": changed,
                "current": self.current(),
                "ignored": False,
            }

    def list_recent(
        self,
        *,
        limit: int = 20,
        since_hours: float | None = None,
        now: float | None = None,
    ) -> list:
        with self._lock:
            values = list(self.segments)
            if since_hours is not None:
                cutoff = float(now if now is not None else _time.time()) - max(
                    0.0, float(since_hours)
                ) * 3600
                values = [
                    item for item in values
                    if item.get("ended_at") is None
                    or float(item.get("ended_at", 0)) >= cutoff
                ]
            return copy.deepcopy(values[-max(1, min(int(limit), 100)):])
