"""Gemma4 API client — tool-calling support via llama.cpp OpenAI-compatible endpoint."""

import json
import requests
from config import (
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_MAX_TOKENS,
    LLM_MAX_TOKENS_COMPACT,
    LLM_TIMEOUT,
)


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

        return {
            "content": (msg.get("content") or "").strip(),
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
