"""Conversation context manager with timestamped messages and message-count-based compaction."""

import json
import os
import time as _time
from config import COMPACT_AFTER_N_MESSAGES, KEEP_LAST_N_MESSAGES, PROJECT_DIR, TOKEN_ESTIMATE_DIVISOR, MERGE_SUMMARIES_AFTER
from logger import info

CONTEXT_FILE = os.path.join(PROJECT_DIR, "context.json")


def _ts_fmt(ts: float) -> str:
    return _time.strftime("[%a %H:%M:%S]", _time.localtime(ts))


def _date_fmt(ts: float) -> str:
    return _time.strftime("%a %Y-%m-%d %H:%M", _time.localtime(ts))


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
            info(f"[CONTEXT] Saved {len(to_save)} messages to {CONTEXT_FILE}")
        except Exception as e:
            info(f"[CONTEXT] Save error: {e}")

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
            info(f"[CONTEXT] Loaded {len(self.messages)} messages from {CONTEXT_FILE}")
            return True
        except Exception as e:
            info(f"[CONTEXT] Load error: {e}")
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
            info(f"[CONTEXT] Repaired {repairs} pairing violations, saving...")
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
            msg = {k: v for k, v in m.items() if not k.startswith("_")}
            ts = m.get("_ts")
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

    @staticmethod
    def _format_tool_result(msg: dict) -> str | None:
        """Extract meaningful content from a tool result for compaction.
        
        Returns None if the result is pure boilerplate and should be skipped entirely.
        Returns a clean, concise text string otherwise.
        """
        name = msg.get("name", "?")
        content = msg.get("content", "")
        if not content:
            return None
            
        try:
            data = json.loads(content) if isinstance(content, str) else content
        except json.JSONDecodeError:
            return content.strip()

        if not isinstance(data, dict):
            return content.strip()

        error = data.get("error", False)
        status = data.get("status", "")

        # --- Boilerplate tools: skip on success, keep on error ---
        if name in ("update_display",):
            return None if not error else f"[display error: {data.get('message', '')}]"

        if name in ("delete_notification",):
            return None if not error else f"[delete failed: {data.get('message', '')}]"

        if name in ("send_chat_message",):
            return None

        if name in ("schedule_notification",):
            return None if not error else f"[schedule failed: {data.get('message', '')}]"

        # --- wait: compact summary ---
        if name == "wait":
            waited = data.get("waited", "?")
            reason = data.get("reason", "")
            if status == "interrupted":
                return f"[waited {waited}s, interrupted: {reason}]"
            return f"[waited {waited}s]"

        # --- Photo tools: extract description only ---
        if name in ("take_photo", "capture_photo"):
            desc = data.get("description", "")
            if desc:
                return f"[scene] {desc.strip()}"
            return None

        # --- Search tools: extract titles and snippets ---
        if name in ("brave_web_search", "brave_local_search"):
            results = data.get("results", [])
            if not results:
                title = data.get("title", "") or data.get("query", "")
                snippet = data.get("description", "") or data.get("extra_snippets", "")
                if isinstance(snippet, list):
                    snippet = " | ".join(snippet[:2])
                if title:
                    snippet_str = f": {snippet}" if snippet else ""
                    return f"[search] {title}{snippet_str}"
                return f"[search: no results]"
            lines = []
            for r in results[:8]:
                t = r.get("title", "") if isinstance(r, dict) else str(r)
                s = (r.get("description", "") or r.get("extra_snippets", [])) if isinstance(r, dict) else ""
                if isinstance(s, list):
                    s = " | ".join(s[:2])
                if s:
                    lines.append(f"- {t}: {s}")
                else:
                    lines.append(f"- {t}")
            return f"[search results]\n" + "\n".join(lines)

        if name == "brave_news_search":
            results = data.get("results", [])
            if not results:
                return f"[news: no results]"
            lines = []
            for r in results[:8]:
                if isinstance(r, dict):
                    lines.append(f"- {r.get('title', '')}: {r.get('description', '')}")
                else:
                    lines.append(f"- {r}")
            return f"[news]\n" + "\n".join(lines)

        if name in ("brave_image_search", "brave_video_search"):
            results = data.get("results", [])
            if not results:
                return f"[{name}: no results]"
            titles = [r.get("title", "") if isinstance(r, dict) else str(r) for r in results[:5]]
            return f"[{name}: {', '.join(titles)}]"

        if name == "brave_summarizer":
            summary = data.get("summary", "") or data.get("content", "")
            if summary:
                return f"[summary] {str(summary).strip()}"
            return None

        # --- Vision requests ---
        if name == "update_vision_requests":
            requests_text = data.get("requests", "") or data.get("message", "")
            if requests_text:
                return f"[vision: {str(requests_text).strip()}]"
            return None

        # --- Notification proposals: keep details ---
        if name == "propose_notification":
            notif = data.get("notification", data)
            title = notif.get("title", "") or notif.get("category", "")
            desc = notif.get("description", "") or notif.get("message", "")
            if title or desc:
                return f"[proposed: {title} — {desc}]"
            return f"[proposed notification]"

        # --- Fallback: simplify JSON to key fields ---
        meaningful = {}
        for k, v in data.items():
            if k in ("status", "error", "message"):
                continue
            if isinstance(v, str) and len(v) < 200:
                meaningful[k] = v
            elif isinstance(v, str):
                meaningful[k] = v[:200] + "..."
            elif isinstance(v, (int, float, bool)):
                meaningful[k] = v
        if meaningful:
            return json.dumps(meaningful)
        return None

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

    def _prepare_compact(self):
        """Gather data for compaction. Must be called with lock held. Returns (combined, fallback_summary, system_msg, existing_summaries, end, keep_count, to_compact_len, date_label) or None."""
        if len(self.messages) < COMPACT_AFTER_N_MESSAGES:
            return None

        system_msg = self.messages[0] if self.messages and self.messages[0]["role"] == "system" else None
        keep_count = KEEP_LAST_N_MESSAGES

        start = 1 if system_msg else 0
        end = len(self.messages) - keep_count

        if end <= start:
            return None

        end = self._find_safe_end(start, end)
        if end <= start:
            return None

        window = self.messages[start:end]

        existing_summaries = [m for m in window if self._is_summary(m)]
        to_compact = [m for m in window if not self._is_summary(m)]

        if not to_compact:
            return None

        ts_min = min(m.get("_ts", self._now()) for m in to_compact)
        ts_max = max(m.get("_ts", self._now()) for m in to_compact)
        date_label = f"{_date_fmt(ts_min)} \u2013 {_date_fmt(ts_max)}"

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
                content = f"[{tools}] {content}"
            if role == "tool":
                compacted = self._format_tool_result(m)
                if compacted is None:
                    continue
                content = compacted
                role = "tool"
            text_parts.append(f"{ts_str} {role}: {content}")

        combined = "\n".join(text_parts)

        # Fallback summary if the LLM call fails: preserve raw user messages
        # so they aren't lost when the window is discarded.
        user_msgs = []
        for m in to_compact:
            if m.get("role") == "user" and not m.get("tool_calls"):
                content = m.get("content", "")
                if isinstance(content, list):
                    content = " ".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
                if content and isinstance(content, str) and len(content.strip()) > 0:
                    user_msgs.append(f"{_ts_fmt(m.get('_ts', 0))} {content.strip()[:300]}")
        if user_msgs:
            fallback_summary = "COMPACTION FAILED — only user messages preserved:\n" + "\n".join(user_msgs[:30])
        else:
            fallback_summary = "Compaction failed for this period — no recoverable content."

        return (combined, fallback_summary, system_msg, existing_summaries, end, keep_count, len(to_compact), date_label, ts_min, ts_max)

    def _apply_compact(self, system_msg, existing_summaries, end, keep_count, to_compact_len, date_label, summary, ts_min, ts_max):
        """Apply compaction result. Must be called with lock held."""
        new_messages = []
        if system_msg:
            new_messages.append(system_msg)
        new_messages.extend(existing_summaries)
        new_messages.append({
            "role": "user",
            "content": f"[Previous context summary: {date_label}] {summary}",
            "_ts": self._now(),
            "_ts_min": ts_min,
            "_ts_max": ts_max,
        })
        new_messages.extend(self.messages[end:])
        self.messages = new_messages
        info(f"[CONTEXT] Compacted {to_compact_len} messages into 1 summary (~{len(summary)} chars) [{date_label}], "
              f"preserved {len(existing_summaries)} prior summaries, keeping last {keep_count}")

    def check_compact(self, ai_client, ctx_lock=None):
        """Compact context. Does not hold lock during LLM call."""
        if ctx_lock:
            with ctx_lock:
                prep = self._prepare_compact()
        else:
            prep = self._prepare_compact()

        if prep is None:
            return

        combined, fallback_summary, system_msg, existing_summaries, end, keep_count, to_compact_len, date_label, ts_min, ts_max = prep

        try:
            summary = ai_client.compact(combined)
        except Exception as e:
            info(f"[CONTEXT] Compaction LLM call failed: {e}")
            summary = fallback_summary

        if ctx_lock:
            with ctx_lock:
                self._apply_compact(system_msg, existing_summaries, end, keep_count, to_compact_len, date_label, summary, ts_min, ts_max)
        else:
            self._apply_compact(system_msg, existing_summaries, end, keep_count, to_compact_len, date_label, summary, ts_min, ts_max)

    def _prepare_merge_summaries(self):
        """Gather data for merge. Must be called with lock held. Returns (summaries_text, ranges) or None.

        Summaries are numbered [1]..[N] so the LLM can report which inputs each
        merged entry covers; ranges[i] holds the i-th summary's (_ts_min, _ts_max)
        so merged date labels can be computed in code instead of trusted to the LLM.
        """
        summary_items = [(i, m) for i, m in enumerate(self.messages) if self._is_summary(m)]
        if len(summary_items) <= MERGE_SUMMARIES_AFTER:
            return None
        ranges = [(m.get("_ts_min"), m.get("_ts_max")) for _, m in summary_items]
        summaries_text = "\n\n---\n\n".join(f"[{n}] {m['content']}" for n, (_, m) in enumerate(summary_items, 1))
        info(f"[CONTEXT] Merging {len(summary_items)} summaries ({len(summaries_text)} total chars)...")
        return summaries_text, ranges

    def _merged_range(self, item, ranges, kept_idxs):
        """Compute (ts_min, ts_max) for a merged entry from its source summaries' stored ranges. Returns None if unavailable (bad/missing sources, legacy summaries without _ts_min/_ts_max, or a kept summary falling inside the span — a min/max label would falsely claim coverage of that period)."""
        sources = item.get("sources")
        if not isinstance(sources, list) or not sources:
            return None
        idxs = []
        for s in sources:
            try:
                idxs.append(int(s))
            except (TypeError, ValueError):
                return None
        if any(i < 1 or i > len(ranges) for i in idxs):
            return None
        if any(i in kept_idxs and i not in idxs for i in range(min(idxs) + 1, max(idxs))):
            return None
        mins = [ranges[i - 1][0] for i in idxs]
        maxs = [ranges[i - 1][1] for i in idxs]
        if any(v is None for v in mins) or any(v is None for v in maxs):
            return None
        return min(mins), max(maxs)

    def _apply_merge_summaries(self, summaries_text, ranges, merged):
        """Apply merge result. Must be called with lock held."""
        if not merged:
            info(f"[CONTEXT] Summary merge produced no summaries, skipping. Input was {len(summaries_text)} chars")
            return

        summary_items = [(i, m) for i, m in enumerate(self.messages) if self._is_summary(m)]
        if len(merged) >= len(summary_items):
            info(f"[CONTEXT] Summary merge produced {len(merged)} summaries (was {len(summary_items)}), skipping — not an improvement")
            return

        kept_idxs = set()
        for item in merged:
            if isinstance(item, dict):
                for s in item.get("sources") or []:
                    try:
                        kept_idxs.add(int(s))
                    except (TypeError, ValueError):
                        pass

        new_summaries = []
        for item in merged:
            if not isinstance(item, dict) or "summary" not in item:
                continue
            span = self._merged_range(item, ranges, kept_idxs)
            if span:
                ts_min, ts_max = span
                label = f"{_date_fmt(ts_min)} – {_date_fmt(ts_max)}"
            elif item.get("time_range"):
                label = str(item["time_range"])
            else:
                continue
            msg = {
                "role": "user",
                "content": f"[Previous context summary: {label}] {item['summary']}",
                "_ts": self._now(),
            }
            if span:
                msg["_ts_min"], msg["_ts_max"] = span
            new_summaries.append(msg)

        if not new_summaries or len(new_summaries) >= len(summary_items):
            info(f"[CONTEXT] Summary merge result invalid ({len(new_summaries)} valid summaries), skipping")
            return

        non_summaries = [m for m in self.messages if not self._is_summary(m)]
        insert_at = 1 if non_summaries and non_summaries[0].get("role") == "system" else 0
        non_summaries[insert_at:insert_at] = new_summaries
        self.messages = non_summaries
        info(f"[CONTEXT] Merged {len(summary_items)} summaries into {len(new_summaries)}")
        self.save()

    def check_merge_summaries(self, ai_client, ctx_lock=None):
        """Merge, condense, or drop old compaction summaries. Only touches summary messages. Does not hold lock during LLM call."""
        if ctx_lock:
            with ctx_lock:
                prep = self._prepare_merge_summaries()
        else:
            prep = self._prepare_merge_summaries()

        if prep is None:
            return
        summaries_text, ranges = prep

        try:
            merged = ai_client.merge_summaries(summaries_text)
        except Exception as e:
            info(f"[CONTEXT] Summary merge error: {e}")
            return

        if ctx_lock:
            with ctx_lock:
                self._apply_merge_summaries(summaries_text, ranges, merged)
        else:
            self._apply_merge_summaries(summaries_text, ranges, merged)

    def _is_image_message(self, msg: dict) -> bool:
        content = msg.get("content", "")
        if isinstance(content, list):
            return any(
                isinstance(p, dict) and p.get("type") == "image_url"
                for p in content
            )
        return False
