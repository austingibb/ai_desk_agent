"""Conversation context manager with token estimation and automatic compaction."""

import json
from config import MAX_CONTEXT_TOKENS, KEEP_LAST_N_MESSAGES


class Context:
    def __init__(self):
        self.messages = []

    def add_system(self, content: str):
        self.messages.append({"role": "system", "content": content})

    def add_user(self, content: str):
        self.messages.append({"role": "user", "content": content})

    def add_image(self, photo_uri: str):
        self.messages = [m for m in self.messages if not self._is_image_message(m)]
        self.messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Here is the latest photo from the camera."},
                {"type": "image_url", "image_url": {"url": photo_uri}},
            ],
        })

    def add_assistant(self, response: dict):
        msg = {"role": "assistant"}
        content = response.get("content", "")
        if content:
            msg["content"] = content
        tool_calls = response.get("tool_calls", [])
        if tool_calls:
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
            if "content" not in msg:
                msg["content"] = ""
        self.messages.append(msg)

    def add_tool_result(self, tool_call_id: str, name: str, result: dict):
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name,
            "content": json.dumps(result),
        })

    def get_messages(self) -> list:
        return list(self.messages)

    def total_tokens(self) -> int:
        total = 0
        for msg in self.messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        total += 3500
                    elif isinstance(part, dict) and part.get("type") == "text":
                        total += len(part.get("text", "")) // 4
            elif isinstance(content, str):
                total += len(content) // 4
            for tc in msg.get("tool_calls", []):
                total += len(json.dumps(tc)) // 4
        return total

    def check_compact(self, ai_client):
        if self.total_tokens() < MAX_CONTEXT_TOKENS * 0.7:
            return

        system_msg = self.messages[0] if self.messages and self.messages[0]["role"] == "system" else None
        keep_count = KEEP_LAST_N_MESSAGES

        if len(self.messages) <= keep_count + (1 if system_msg else 0):
            return

        start = 1 if system_msg else 0
        end = len(self.messages) - keep_count
        to_compact = self.messages[start:end]

        text_parts = []
        for m in to_compact:
            role = m.get("role", "?")
            content = m.get("content", "")
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
            text_parts.append(f"{role}: {content}")

        combined = "\n".join(text_parts)
        try:
            summary = ai_client.compact(combined)
        except Exception:
            summary = combined[:1000]

        new_messages = []
        if system_msg:
            new_messages.append(system_msg)
        new_messages.append({"role": "user", "content": f"[Previous context summary: {summary}]"})
        new_messages.extend(self.messages[end:])
        self.messages = new_messages

    def _is_image_message(self, msg: dict) -> bool:
        content = msg.get("content", "")
        if isinstance(content, list):
            return any(
                isinstance(p, dict) and p.get("type") == "image_url"
                for p in content
            )
        return False
