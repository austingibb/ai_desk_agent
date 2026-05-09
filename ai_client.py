"""LLM API clients — DeepSeek on OpenRouter (brain) + local Gemma (vision/compaction)."""

import json
import re
import requests
from config import (
    LLM_BASE_URL,
    LLM_API_KEY,
    LLM_MODEL,
    VISION_BASE_URL,
    VISION_MODEL,
    VISION_API_KEY,
    VISION_PROMPT,
    VISION_TIMEOUT,
    LLM_MAX_TOKENS,
    LLM_MAX_TOKENS_COMPACT,
    LLM_TIMEOUT,
)

# Gemma4 can leak control tokens like <|"|> into output
_TOKEN_JUNK_RE = re.compile(r"<\|[\"']{1,3}\|?>")


def _clean(text: str) -> str:
    return _TOKEN_JUNK_RE.sub("", text).strip()


class AIClient:
    """Brain LLM — DeepSeek on OpenRouter for reasoning and tool calling."""

    def __init__(self):
        self.base_url = LLM_BASE_URL.rstrip("/")
        self.model = LLM_MODEL
        self._headers = {
            "HTTP-Referer": "https://github.com/ai-eink-friend",
            "X-Title": "AI E-Ink Friend",
        }
        if LLM_API_KEY:
            self._headers["Authorization"] = f"Bearer {LLM_API_KEY}"
        self._vision = VisionClient()

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
            headers=self._headers,
            json=payload,
            timeout=LLM_TIMEOUT,
        )
        if not resp.ok:
            body = resp.text[:500]
            print(f"[LLM] {resp.status_code} error: {body}")
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
        return {
            "content": content,
            "reasoning": (msg.get("reasoning_content") or "").strip(),
            "tool_calls": tool_calls,
            "raw_message": msg,
        }

    def compact(self, text: str) -> str:
        """Summarize old context using the local Gemma model (free)."""
        return self._vision.compact(text)


class VisionClient:
    """Local Gemma on llama.cpp — vision descriptions and compaction."""

    def __init__(self):
        self.base_url = VISION_BASE_URL.rstrip("/")
        self.model = VISION_MODEL
        self._headers = {}
        if VISION_API_KEY:
            self._headers["Authorization"] = f"Bearer {VISION_API_KEY}"

    def describe(self, image_data_uri: str) -> str:
        """Send a photo to local Gemma and get a text description."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": VISION_PROMPT},
                    {"type": "image_url", "image_url": {"url": image_data_uri}},
                ],
            }
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 512,
            "temperature": 0.3,
        }
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers,
            json=payload,
            timeout=VISION_TIMEOUT,
        )
        resp.raise_for_status()
        raw = resp.json()["choices"][0]["message"]["content"]
        return _clean(raw)

    def compact(self, text: str) -> str:
        """Summarize old context using local Gemma (free)."""
        messages = [
            {"role": "user", "content": f"Summarize these observations and interactions concisely, preserving key events, decisions, and patterns. Pay attention to timestamps to understand the sequence and timing of events:\n\n{text}"}
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": LLM_MAX_TOKENS_COMPACT,
            "temperature": 0.3,
        }
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers,
            json=payload,
            timeout=LLM_TIMEOUT,
        )
        resp.raise_for_status()
        return _clean(resp.json()["choices"][0]["message"]["content"])
