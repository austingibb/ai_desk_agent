"""Conversation context manager with token estimation and automatic compaction."""

import time
import json
from config import MAX_CONTEXT_TOKENS, KEEP_LAST_N_EXCHANGES, COMPACT_PROMPT


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    return len(text) // 4


def estimate_image_tokens(width: int, height: int) -> int:
    """Rough estimate for base64 image tokens in llama.cpp multimodal."""
    pixels = width * height
    return (pixels // 100) + 100


class Context:
    def __init__(self):
        self.messages = []
        self.window_reasoning = []
        self.previous_displays = []
        self.last_photo_tokens = 0
        self.photo_width = 0
        self.photo_height = 0
        self.total_photos = 0

    def add_system(self, content: str):
        self.messages.append({"role": "system", "content": content})

    def add_reasoning(self, reasoning: str):
        self.window_reasoning.append(
            {"timestamp": time.strftime("%I:%M %p"), "reasoning": reasoning}
        )

    def add_display(self, text: str):
        self.previous_displays.append(text)
        if len(self.previous_displays) > 10:
            self.previous_displays = self.previous_displays[-10:]

    def add_button_response(self, question: str, answer: str):
        text = f"User responded to '{question}' with: {answer}"
        self.window_reasoning.append({"timestamp": time.strftime("%I:%M %p"), "reasoning": text})

    def get_window_text(self) -> str:
        lines = []
        for entry in self.window_reasoning:
            lines.append(f"[{entry['timestamp']}] {entry['reasoning']}")
        return "\n".join(lines)

    def get_previous_display(self) -> str:
        if self.previous_displays:
            return self.previous_displays[-1]
        return "(none — first display)"

    def close_window(self, ai_client) -> str:
        """Compact current window observations into a summary. Returns summary text."""
        observations = self.get_window_text()
        if not observations.strip():
            self.window_reasoning = []
            return ""

        prompt = COMPACT_PROMPT.format(observations=observations)

        try:
            summary = ai_client._simple_chat(prompt, system=None)
            compacted = f"[Window summary: {summary}]"
        except Exception:
            compacted = f"[Window observations: {observations[:500]}...]"

        self.messages.append({"role": "user", "content": compacted})
        self.window_reasoning = []
        return summary

    def build_request_data(self, photo_base64: str, photo_width: int, photo_height: int) -> list:
        self.photo_width = photo_width
        self.photo_height = photo_height
        self.last_photo_tokens = estimate_image_tokens(photo_width, photo_height)
        self.total_photos += 1

        merged = list(self.messages)

        if not merged or merged[0].get("role") != "system":
            merged.insert(0, {"role": "system", "content": ""})

        reasoning_text = self.get_window_text()
        if reasoning_text:
            text = (
                f"Here is the latest photo of the room. "
                f"Your observations so far this window:\n{reasoning_text}\n\n"
                f"Observe this new photo and add your thoughts."
            )
        elif self.total_photos == 1:
            text = "Here is the first photo of the room. What do you observe?"
        else:
            text = "Here is the first photo of a new observation window. What do you observe?"

        merged.append({
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": photo_base64}},
            ],
        })

        return merged

    def total_tokens(self) -> int:
        total = 0
        for msg in self.messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        total += self.last_photo_tokens
                    elif isinstance(part, str):
                        total += estimate_tokens(part)
                    elif isinstance(part, dict):
                        total += estimate_tokens(str(part))
            else:
                total += estimate_tokens(str(content))
        total += estimate_tokens(self.get_window_text())
        return total

    def check_compact(self, ai_client):
        """Compact if approaching token limit."""
        current = self.total_tokens()
        if current < MAX_CONTEXT_TOKENS * 0.7:
            return

        system_msg = self.messages[0] if self.messages and self.messages[0]["role"] == "system" else None
        to_keep = self.messages[-KEEP_LAST_N_EXCHANGES:] if len(self.messages) > KEEP_LAST_N_EXCHANGES else self.messages

        middle = self.messages[1:-KEEP_LAST_N_EXCHANGES] if system_msg else self.messages[:-KEEP_LAST_N_EXCHANGES]
        if not middle:
            return

        combined = "\n".join(str(m.get("content", "")) for m in middle)
        if not combined.strip():
            return

        try:
            prompt = f"Summarize this conversation history concisely:\n\n{combined}"
            summary = ai_client._simple_chat(prompt, system="Summarize concisely.")
        except Exception:
            summary = combined[:1000] + "..."

        new_messages = [system_msg] if system_msg else []
        new_messages.append({"role": "user", "content": f"[Compacted history: {summary}]"})
        new_messages.extend(to_keep)
        self.messages = new_messages
