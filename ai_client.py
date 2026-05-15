"""LLM API clients — DeepSeek on OpenRouter (brain) + local Gemma (vision)."""

import json
import re
import requests
from logger import info
from config import (
    LLM_BASE_URL,
    LLM_API_KEY,
    LLM_MODEL,
    VISION_BASE_URL,
    VISION_MODEL,
    VISION_API_KEY,
    VISION_PROMPT_BASE,
    VISION_REQUESTS_FILE,
    VISION_TIMEOUT,
    LLM_MAX_TOKENS,
    LLM_MAX_TOKENS_COMPACT,
    LLM_TIMEOUT,
    MERGE_SUMMARIES_TARGET,
    TOKEN_ESTIMATE_DIVISOR,
)

# Gemma4 can leak control tokens like <|"|> into output
_TOKEN_JUNK_RE = re.compile(r"<\|[\"']{1,3}\|?>")

_PURPOSE = (
    "Your real purpose is keeping Austin honest about the daily stuff — "
    "getting up from the desk, drinking water, staying on track with studying "
    "and applications instead of drifting. You're the small nudge in the moment, "
    "the reminder of what he said he wanted, so the long-term goals actually get "
    "there one day at a time. On the health habits that matter, you're firm — "
    "you keep asking until he actually moves."
)


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
            info(f"[LLM] {resp.status_code} error: {body}")
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
        """Summarize old context using DeepSeek."""
        messages = [
            {"role": "user", "content": f"{_PURPOSE} When summarizing, prioritize information that helps you fulfill this role.\n\nSummarize these observations and interactions concisely, preserving key events, decisions, and patterns. Pay attention to timestamps to understand the sequence and timing of events:\n\n{text}"}
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
        return resp.json()["choices"][0]["message"]["content"].strip()

    def merge_summaries(self, summaries_text: str) -> str:
        """Merge/condense/drop context summaries using DeepSeek."""
        info(f"[LLM] merge_summaries: {len(summaries_text)} chars input, targeting <= {MERGE_SUMMARIES_TARGET} summaries")
        prompt = (
            f"You are reviewing a series of context summaries from an AI assistant's conversation history.\n"
            f"Each summary was created at a different time and covers a different period.\n\n"
            f"Your job is to REDUCE the number of summaries to at most {MERGE_SUMMARIES_TARGET} while preserving the most important information. You can:\n"
            f"- MERGE related or adjacent summaries into one\n"
            f"- CONDENSE summaries that are too detailed (e.g., drop repetitive photo/wait cycles)\n"
            f"- DROP summaries that contain only routine monitoring with no meaningful events\n"
            f"- KEEP important summaries as-is\n\n"
            f"Remember your core purpose when deciding what to keep: {_PURPOSE}\n\n"
            f"Prioritize preserving:\n"
            f"- User preferences, corrections, and personality details\n"
            f"- Key decisions and their reasons\n"
            f"- Important events (notifications created, topics discussed, user habits learned)\n"
            f"- Emotional/relationship moments\n"
            f"- Anything that helps you fulfill your role as a regulation partner (patterns of behavior, triggers, what works)\n\n"
            f"Deprioritize:\n"
            f"- Repetitive photo descriptions of the same room\n"
            f"- Routine wait/display/photo tool cycles\n"
            f"- Redundant restatements of the same information across summaries\n\n"
            f"Return a JSON array of objects. Each object has:\n"
            f'- "time_range": the time period covered (e.g., "[Sat 14:33:40] \\u2013 [Sat 16:04:37]")\n'
            f'- "summary": the summary text\n\n'
            f"Return ONLY the JSON array, no other text.\n\n"
            f"Here are the current summaries:\n\n"
            f"{summaries_text}"
        )
        max_tokens = 8192  # DeepSeek max output
        info(f"[LLM] merge_summaries: prompt={len(prompt)} chars, sending...")
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.3,
        }
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers,
            json=payload,
            timeout=LLM_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        finish = choice.get("finish_reason", "unknown")
        content = choice["message"].get("content")
        if not content:
            info(f"[LLM] merge_summaries: empty content, finish_reason={finish}")
            return ""
        result = content.strip()
        if finish == "length":
            info(f"[LLM] merge_summaries: hit max_tokens ({max_tokens}) — result may be truncated")
        info(f"[LLM] merge_summaries: got {len(result)} chars response")
        return result


class VisionClient:
    """Local Gemma on llama.cpp — vision descriptions."""

    def __init__(self):
        self.base_url = VISION_BASE_URL.rstrip("/")
        self.model = VISION_MODEL
        self._headers = {}
        if VISION_API_KEY:
            self._headers["Authorization"] = f"Bearer {VISION_API_KEY}"

    def _build_vision_prompt(self) -> str:
        """Build vision prompt from base + requests file."""
        prompt = VISION_PROMPT_BASE
        try:
            with open(VISION_REQUESTS_FILE, "r") as f:
                extra = f.read().strip()
            if extra:
                prompt += "\n\n" + extra
        except FileNotFoundError:
            pass
        return prompt

    def describe(self, image_data_uri: str, max_retries: int = 3) -> str:
        """Send a photo to local Gemma and get a text description. Retries on empty."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self._build_vision_prompt()},
                    {"type": "image_url", "image_url": {"url": image_data_uri}},
                ],
            }
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 2048,
            "temperature": 0.3,
        }
        for attempt in range(max_retries):
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers,
                json=payload,
                timeout=VISION_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            finish = choice.get("finish_reason", "unknown")
            raw = choice["message"]["content"]
            cleaned = _clean(raw)
            if cleaned:
                if finish == "length":
                    info(f"[VISION] Description truncated (hit max_tokens). Length: {len(cleaned)} chars")
                return cleaned
            info(f"[VISION] Empty response (attempt {attempt + 1}/{max_retries}). Raw: {repr(raw[:200])}")
        return ""
