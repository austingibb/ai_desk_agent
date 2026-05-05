"""Gemma4 API client — tool-calling support via llama.cpp OpenAI-compatible endpoint."""

import json
import re
import requests
from config import (
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_MAX_TOKENS,
    LLM_MAX_TOKENS_COMPACT,
    LLM_TIMEOUT,
)

# Gemma4 on llama.cpp sometimes outputs tool calls as inline text
# e.g. <|tool_call>call:wait{seconds:600}<tool_call|>
INLINE_TOOL_RE = re.compile(
    r"<\|tool_call>\s*call:(\w+)\s*(\{[^}]*\})\s*<tool_call\|?>",
    re.DOTALL,
)


def _parse_inline_args(args_str: str) -> dict:
    """Parse {key:value, ...} from inline tool call (non-JSON, unquoted keys)."""
    result = {}
    if not args_str.strip():
        return result
    inner = args_str.strip().strip("{}")
    pairs = [p.strip() for p in inner.split(",") if p.strip()]
    for pair in pairs:
        if ":" not in pair:
            continue
        key, val = pair.split(":", 1)
        key = key.strip()
        val = val.strip()
        if val.isdigit():
            val = int(val)
        elif val.lower() in ("true", "false"):
            val = val.lower() == "true"
        else:
            val = val.strip('"').strip("'")
        result[key] = val
    return result


class AIClient:
    def __init__(self):
        self.base_url = LLM_BASE_URL.rstrip("/")
        self.model = LLM_MODEL

    def chat_with_tools(self, messages: list, tools: list = None) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": LLM_MAX_TOKENS,
            "temperature": 0.7,
        }
        if tools:
            payload["tools"] = tools

        resp = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            timeout=LLM_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        msg = choice.get("message", {})

        tool_calls = []
        for tc in msg.get("tool_calls", []):
            args = tc["function"]["arguments"]
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            tool_calls.append({
                "id": tc.get("id", f"call_{len(tool_calls)}"),
                "name": tc["function"]["name"],
                "arguments": args,
            })

        content = (msg.get("content") or "").strip()

        if not tool_calls:
            for m in INLINE_TOOL_RE.finditer(content):
                name = m.group(1)
                args = _parse_inline_args(m.group(2))
                tool_calls.append({
                    "id": f"call_{len(tool_calls)}",
                    "name": name,
                    "arguments": args,
                })
            if tool_calls:
                content = INLINE_TOOL_RE.sub("", content).strip()

        return {
            "content": content,
            "reasoning": (msg.get("reasoning_content") or "").strip(),
            "tool_calls": tool_calls,
            "raw_message": msg,
        }

    def compact(self, text: str) -> str:
        messages = [
            {"role": "user", "content": f"Summarize these observations and interactions concisely, preserving key events and patterns:\n\n{text}"}
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": LLM_MAX_TOKENS_COMPACT,
            "temperature": 0.3,
        }
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            timeout=LLM_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
