"""Conversation context manager with timestamped messages and message-count-based compaction."""

import json
import os
import time as _time
from config import COMPACT_AFTER_N_MESSAGES, KEEP_LAST_N_MESSAGES, PROJECT_DIR, TOKEN_ESTIMATE_DIVISOR

CONTEXT_FILE = os.path.join(PROJECT_DIR, "context.json")


def _ts_fmt(ts: float) -> str:
    return _time.strftime("[%a %H:%M:%S]", _time.localtime(ts))


class Context:
    def __init__(self):
        self.messages = []

    def save(self):
        """Save messages to disk, stripping image data."""
        to_save = []
        for m in self.messages:
            if self._is_image_message(m):
                continue
            to_save.append(m)
        try:
            with open(CONTEXT_FILE, "w") as f:
                json.dump(to_save, f)
            print(f"[CONTEXT] Saved {len(to_save)} messages to {CONTEXT_FILE}")
        except Exception as e:
            print(f"[CONTEXT] Save error: {e}")

    def load(self) -> bool:
        """Load messages from disk. Returns True if loaded successfully."""
        if not os.path.exists(CONTEXT_FILE):
            return False
        try:
            with open(CONTEXT_FILE, "r") as f:
                self.messages = json.load(f)
            now = _time.time()
            for m in self.messages:
                if "_ts" not in m:
                    m["_ts"] = now
            repaired = self._repair_pairing()
            print(f"[CONTEXT] Loaded {len(self.messages)} messages from {CONTEXT_FILE}")
            return True
        except Exception as e:
            print(f"[CONTEXT] Load error: {e}")
            return False

    def _repair_pairing(self) -> int:
        """Remove messages that violate OpenAI assistant/tool pairing rules.

        Three violations fixed:
        1. Tool messages without a preceding assistant tool_calls entry (orphan tools)
        2. Non-tool messages sandwiched between an assistant with tool_calls
           and its tool results (OpenRouter requires tool to immediately follow assistant)
        3. Assistant tool_calls with no matching tool result (trimmed from tool_calls list;
           if all are unmatched, the assistant message is kept without tool_calls)

        Returns the number of repairs made.
        """
        repairs = 0
        pending_ids = set()
        current_assistant = None  # track assistant with pending tool_calls
        fulfilled_ids = set()    # tool_call IDs that got results
        cleaned = []

        def _trim_assistant(assistant_msg, fulfilled):
            """Remove unfulfilled tool_calls from an assistant message."""
            nonlocal repairs
            original = assistant_msg.get("tool_calls", [])
            kept = [tc for tc in original if tc["id"] in fulfilled]
            if len(kept) < len(original):
                repairs += len(original) - len(kept)
                if kept:
                    assistant_msg["tool_calls"] = kept
                else:
                    del assistant_msg["tool_calls"]

        for m in self.messages:
            role = m.get("role", "")
            if role == "assistant" and m.get("tool_calls"):
                # Trim previous assistant if it had unfulfilled tool_calls
                if current_assistant and pending_ids:
                    _trim_assistant(current_assistant, fulfilled_ids)
                pending_ids = {tc["id"] for tc in m["tool_calls"]}
                fulfilled_ids = set()
                current_assistant = m
                cleaned.append(m)
            elif role == "tool":
                tid = m.get("tool_call_id", "")
                if pending_ids and tid in pending_ids:
                    pending_ids.discard(tid)
                    fulfilled_ids.add(tid)
                    cleaned.append(m)
                    if not pending_ids:
                        current_assistant = None
                else:
                    repairs += 1
            elif pending_ids:
                repairs += 1
            else:
                cleaned.append(m)

        # Handle unfulfilled tool_calls on the last assistant message
        if current_assistant and pending_ids:
            _trim_assistant(current_assistant, fulfilled_ids)

        if repairs:
            self.messages = cleaned
            print(f"[CONTEXT] Repaired {repairs} pairing violations, saving...")
            self.save()
        return repairs

    def _now(self) -> float:
        return _time.time()

    def add_system(self, content: str):
        self.messages.append({"role": "system", "content": content, "_ts": self._now()})

    def add_user(self, content: str):
        self.messages.append({"role": "user", "content": content, "_ts": self._now()})

    def add_image(self, photo_uri: str):
        self.messages = [m for m in self.messages if not self._is_image_message(m)]
        self.messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Here is the latest photo from the camera."},
                {"type": "image_url", "image_url": {"url": photo_uri}},
            ],
            "_ts": self._now(),
        })

    def add_assistant(self, response: dict):
        tool_calls = response.get("tool_calls", [])
        content = (response.get("content") or "").strip()

        if not tool_calls:
            return

        msg = {"role": "assistant", "_ts": self._now()}
        if content:
            msg["content"] = content
        else:
            msg["content"] = ""
        msg["tool_calls"] = [
            {
                "id": tc["id"],
                "type": "function",
                "function": {
                    "name": tc["name"],
                    "arguments": json.dumps(tc["arguments"]),
                },
            }
            for tc in tool_calls
        ]
        self.messages.append(msg)

    def add_tool_result(self, tool_call_id: str, name: str, result: dict):
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name,
            "content": json.dumps(result),
            "_ts": self._now(),
        })

    def get_messages(self) -> list:
        result = []
        for m in self.messages:
            msg = dict(m)
            ts = msg.pop("_ts", None)
            if ts and msg.get("role") != "system":
                ts_str = _ts_fmt(ts)
                content = msg.get("content", "")
                if isinstance(content, str):
                    msg["content"] = f"{ts_str} {content}"
                elif isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            part["text"] = f"{ts_str} {part['text']}"
                            break
            result.append(msg)
        return result

    def total_tokens(self) -> int:
        total = 0
        for msg in self.messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        total += 3500
                    elif isinstance(part, dict) and part.get("type") == "text":
                        total += max(1, len(part.get("text", "")) // TOKEN_ESTIMATE_DIVISOR)
            elif isinstance(content, str):
                total += max(1, len(content) // TOKEN_ESTIMATE_DIVISOR)
            for tc in msg.get("tool_calls", []):
                total += max(1, len(json.dumps(tc)) // TOKEN_ESTIMATE_DIVISOR)
        return total

    @staticmethod
    def _is_summary(msg: dict) -> bool:
        """Check if a message is an existing compaction summary."""
        content = msg.get("content", "")
        if not isinstance(content, str):
            return False
        return content.startswith("[Previous context summary:")

    def _find_safe_end(self, start: int, desired_end: int) -> int:
        """Adjust compaction end so we never split an assistant/tool group.
        
        OpenAI requires every 'tool' message to immediately follow the
        'assistant' message that contains its matching 'tool_calls' entry.
        Walk backward from desired_end to find a safe boundary.
        """
        i = desired_end
        while i < len(self.messages):
            msg = self.messages[i]
            if msg.get("role") == "tool":
                i += 1
                continue
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                i += 1
                continue
            break
        return i

    def check_compact(self, ai_client):
        if len(self.messages) < COMPACT_AFTER_N_MESSAGES:
            return

        system_msg = self.messages[0] if self.messages and self.messages[0]["role"] == "system" else None
        keep_count = KEEP_LAST_N_MESSAGES

        start = 1 if system_msg else 0
        end = len(self.messages) - keep_count

        if end <= start:
            return

        end = self._find_safe_end(start, end)
        if end <= start:
            return

        window = self.messages[start:end]

        existing_summaries = [m for m in window if self._is_summary(m)]
        to_compact = [m for m in window if not self._is_summary(m)]

        if not to_compact:
            return

        ts_min = min(m.get("_ts", self._now()) for m in to_compact)
        ts_max = max(m.get("_ts", self._now()) for m in to_compact)
        date_label = f"{_ts_fmt(ts_min)} \u2013 {_ts_fmt(ts_max)}"

        text_parts = []
        for m in to_compact:
            role = m.get("role", "?")
            content = m.get("content", "")
            ts = m.get("_ts", 0)
            ts_str = _ts_fmt(ts)

            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            if m.get("tool_calls"):
                tools = ", ".join(tc["function"]["name"] for tc in m["tool_calls"])
                content = f"[called tools: {tools}] {content}"
            if role == "tool":
                content = f"[tool result for {m.get('name', '?')}] {content}"
            text_parts.append(f"{ts_str} {role}: {content}")

        combined = "\n".join(text_parts)
        try:
            summary = ai_client.compact(combined)
        except Exception:
            summary = combined[:1000]

        new_messages = []
        if system_msg:
            new_messages.append(system_msg)
        new_messages.extend(existing_summaries)
        new_messages.append({
            "role": "user",
            "content": f"[Previous context summary: {date_label}] {summary}",
            "_ts": self._now(),
        })
        new_messages.extend(self.messages[end:])
        self.messages = new_messages
        print(f"[CONTEXT] Compacted {len(to_compact)} messages into 1 summary (~{len(summary)} chars) [{date_label}], "
              f"preserved {len(existing_summaries)} prior summaries, keeping last {keep_count}")

    def _is_image_message(self, msg: dict) -> bool:
        content = msg.get("content", "")
        if isinstance(content, list):
            return any(
                isinstance(p, dict) and p.get("type") == "image_url"
                for p in content
            )
        return False
