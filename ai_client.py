"""Gemma4 API client — multimodal, reasoning-content support."""

import json
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

        result = self._chat(adjusted, max_tokens=LLM_MAX_TOKENS_REASONING)
        parsed = self._parse_reasoning(result)
        return parsed

    def consolidate(self, observations: str, previous_display: str) -> dict:
        prompt = CONSOLIDATE_PROMPT.format(
            observations=observations, previous_display=previous_display
        )
        result = self._chat(
            [{"role": "user", "content": prompt}],
            max_tokens=LLM_MAX_TOKENS_CONSOLIDATE,
            temperature=0.8,
        )
        return self._parse_consolidation(result)

    def _parse_reasoning(self, result: dict) -> dict:
        content = result.get("content", "")
        reasoning = result.get("reasoning", "")

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

        if "DISPLAY:" in content.lower():
            display_part = content.lower().split("display:")[1].strip()
            if display_part.startswith("yes"):
                parsed["should_display"] = True

        if "MESSAGE:" in content:
            msg_parts = content.split("MESSAGE:", 1)
            if len(msg_parts) > 1:
                msg_text = msg_parts[1]
                if "QUESTION:" in msg_text:
                    msg_text = msg_text.split("QUESTION:")[0]
                parsed["display_text"] = msg_text.strip().strip('"').strip("'")

        if "QUESTION:" in content:
            q_parts = content.split("QUESTION:", 1)
            if len(q_parts) > 1:
                parsed["question"] = q_parts[1].strip().strip('"').strip("'")

        return parsed

    def _parse_consolidation(self, result: dict) -> dict:
        content = result.get("content", "")

        parsed = {
            "display_text": content[:200],
            "question": "",
        }

        if "DISPLAY:" in content:
            parts = content.split("DISPLAY:", 1)
            if len(parts) > 1:
                display = parts[1]
                if "ASK:" in display:
                    display = display.split("ASK:")[0]
                parsed["display_text"] = display.strip().strip('"').strip("'")

        if "ASK:" in content:
            parts = content.split("ASK:", 1)
            if len(parts) > 1:
                parsed["question"] = parts[1].strip().strip('"').strip("'")

        return parsed
