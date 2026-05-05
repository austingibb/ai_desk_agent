"""Gemma4 API client — multimodal, reasoning-content support with streaming."""

import json
import sys
import requests
from config import (
    LLM_BASE_URL,
    LLM_MODEL,
    LLM_MAX_TOKENS_REASONING,
    LLM_MAX_TOKENS_CONSOLIDATE,
    LLM_MAX_TOKENS_COMPACT,
    LLM_TIMEOUT,
    CONSOLIDATE_PROMPT,
    SYSTEM_PROMPT,
)


class AIClient:
    def __init__(self):
        self.base_url = LLM_BASE_URL.rstrip("/")
        self.model = LLM_MODEL
        self.system_prompt = SYSTEM_PROMPT

    def _chat(self, messages: list, max_tokens: int, temperature: float = 0.7) -> dict:
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "extra_body": {"reasoning_effort": "low"},
        }
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            timeout=LLM_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        msg = choice.get("message", {})
        return {
            "content": msg.get("content", "").strip(),
            "reasoning": msg.get("reasoning_content", "").strip(),
        }

    def _chat_stream(self, messages: list, max_tokens: int, temperature: float = 0.7):
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "extra_body": {"reasoning_effort": "low"},
        }
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            timeout=LLM_TIMEOUT,
            stream=True,
        )
        resp.raise_for_status()
        return resp

    def _simple_chat(self, user_content: str, system: str = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_content})
        result = self._chat(messages, max_tokens=LLM_MAX_TOKENS_COMPACT, temperature=0.3)
        return result["content"]

    def reason_about_photo(self, messages: list) -> dict:
        prompt = messages[-2]["content"] if len(messages) >= 2 else ""
        has_prior = "Your observations so far this window" in prompt

        if has_prior:
            instruction = (
                "You have just received another photo of the room. "
                "Observe it and add your thoughts. Also consider: do you want to update "
                "the e-ink display now? If yes, include a display message.\n\n"
                "Respond with:\n"
                "REASONING: <your observations and thoughts>\n"
                "DISPLAY: <yes/no>\n"
                "MESSAGE: <display text, ~200 chars max, if DISPLAY is yes>\n"
                "QUESTION: <yes/no question, only if you want to ask one>"
            )
        else:
            instruction = (
                "You have received the first photo of a new observation window. "
                "Observe the room and share your thoughts.\n\n"
                "Respond with:\n"
                "REASONING: <your observations and thoughts>"
            )

        adjusted = list(messages)
        adjusted[-2]["content"] += "\n\n" + instruction

        return self._stream_and_parse(adjusted, LLM_MAX_TOKENS_REASONING)

    def consolidate(self, observations: str, previous_display: str) -> dict:
        prompt = CONSOLIDATE_PROMPT.format(
            observations=observations, previous_display=previous_display
        )
        messages = [{"role": "user", "content": prompt}]
        return self._stream_and_parse(messages, LLM_MAX_TOKENS_CONSOLIDATE)

    def _stream_and_parse(self, messages: list, max_tokens: int) -> dict:
        resp = self._chat_stream(messages, max_tokens, temperature=0.7)
        content = []
        reasoning = []

        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data_str = line[6:]
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    rc = delta.get("reasoning_content", "")
                    ct = delta.get("content", "")
                    if rc:
                        reasoning.append(rc)
                        sys.stdout.write(rc)
                        sys.stdout.flush()
                    if ct:
                        content.append(ct)
                        if not reasoning:
                            sys.stdout.write(ct)
                            sys.stdout.flush()
            except json.JSONDecodeError:
                continue

        full_content = "".join(content)
        full_reasoning = "".join(reasoning)

        return self._parse_raw(full_content, full_reasoning)

    def _parse_raw(self, content: str, reasoning: str) -> dict:
        parsed = {
            "reasoning": reasoning or content[:500],
            "should_display": False,
            "display_text": "",
            "question": "",
        }

        if "REASONING:" in content:
            parts = content.split("REASONING:", 1)
            rest = parts[1] if len(parts) > 1 else content
            parsed["reasoning"] = rest.split("DISPLAY:")[0].strip()

        if "DISPLAY:" in content:
            display_match = content.split("DISPLAY:", 1)
            if len(display_match) > 1:
                rest = display_match[1].split("\n")[0].strip()
                if rest.lower().startswith("yes"):
                    parsed["should_display"] = True
                elif rest.lower() not in ("no", ""):
                    parsed["display_text"] = rest

        if "MESSAGE:" in content:
            msg_parts = content.split("MESSAGE:", 1)
            if len(msg_parts) > 1:
                msg_text = msg_parts[1]
                if "QUESTION:" in msg_text:
                    msg_text = msg_text.split("QUESTION:")[0]
                parsed["display_text"] = msg_text.strip().strip('"').strip("'")

        if "QUESTION:" in content or "ASK:" in content:
            q_key = "QUESTION:" if "QUESTION:" in content else "ASK:"
            q_parts = content.split(q_key, 1)
            if len(q_parts) > 1:
                parsed["question"] = q_parts[1].strip().strip('"').strip("'")

        return parsed
