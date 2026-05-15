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

    def merge_summaries(self, summaries_text: str) -> list:
        """Merge summaries using DeepSeek, collecting output in ~8K chunks to avoid truncation.

        Returns a list of merged summary dicts with 'time_range' and 'summary' keys.
        Empty list on failure.
        """
        info(f"[LLM] merge_summaries: {len(summaries_text)} chars input, targeting <= {MERGE_SUMMARIES_TARGET} summaries")
        accumulated = []
        MAX_CHUNKS = 5

        for chunk_num in range(MAX_CHUNKS):
            is_first = (chunk_num == 0)

            if is_first:
                prompt = _MERGE_PROMPT.format(
                    target=MERGE_SUMMARIES_TARGET,
                    purpose=_PURPOSE,
                    summaries=summaries_text,
                )
            else:
                prompt = _MERGE_CONTINUE.format(
                    target=MERGE_SUMMARIES_TARGET,
                    purpose=_PURPOSE,
                    summaries=summaries_text,
                    produced=json.dumps(accumulated, indent=2),
                )

            content, finish = self._merge_call(prompt)
            info(f"[LLM] merge_summaries chunk {chunk_num + 1}/{MAX_CHUNKS}: {len(content)} chars, finish={finish}")

            if not content:
                if accumulated:
                    break
                continue

            chunk = _parse_json_array(content)
            if chunk is None:
                chunk = _salvage_partial(content)
                if chunk is not None:
                    info(f"[LLM] merge_summaries chunk {chunk_num + 1}: salvaged {len(chunk)} objects from truncated response")

            if chunk:
                accumulated.extend(chunk)
                info(f"[LLM] merge_summaries chunk {chunk_num + 1}: +{len(chunk)} summaries (total {len(accumulated)})")
            else:
                info(f"[LLM] merge_summaries chunk {chunk_num + 1}: unparseable, raw={content[:120]}")
                if accumulated:
                    break
                continue

            if _is_done(content, finish, chunk_num, len(accumulated)):
                break

        info(f"[LLM] merge_summaries: returning {len(accumulated)} summaries after {chunk_num + 1} chunk(s)")
        return accumulated

    def _merge_call(self, prompt: str) -> tuple:
        """Make a merge/completion call. Returns (content, finish_reason)."""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 8192,
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
            info(f"[LLM] _merge_call: empty content, finish_reason={finish}")
            return "", finish
        result = content.strip()
        if finish == "length":
            info(f"[LLM] _merge_call: hit max_tokens — result may be truncated")
        return result, finish


_MERGE_PROMPT = (
    "You are reviewing a series of context summaries from an AI assistant's conversation history.\n"
    "Each summary was created at a different time and covers a different period.\n\n"
    "Your job is to REDUCE the number of summaries to at most {target} while preserving the most important information. You can:\n"
    "- MERGE related or adjacent summaries into one\n"
    "- CONDENSE summaries that are too detailed (e.g., drop repetitive photo/wait cycles)\n"
    "- DROP summaries that contain only routine monitoring with no meaningful events\n"
    "- KEEP important summaries as-is\n\n"
    "Remember your core purpose when deciding what to keep: {purpose}\n\n"
    "Prioritize preserving:\n"
    "- User preferences, corrections, and personality details\n"
    "- Key decisions and their reasons\n"
    "- Important events (notifications created, topics discussed, user habits learned)\n"
    "- Emotional/relationship moments\n"
    "- Anything that helps you fulfill your role as a regulation partner\n\n"
    "Deprioritize:\n"
    "- Repetitive photo descriptions of the same room\n"
    "- Routine wait/display/photo tool cycles\n"
    "- Redundant restatements of the same information across summaries\n\n"
    "Return a JSON array of objects. Each object has:\n"
    '- "time_range": the time period covered\n'
    '- "summary": the summary text\n\n'
    "TARGET: produce at most {target} summaries.\n\n"
    "IMPORTANT — Output in batches to avoid truncation:\n"
    "After your JSON array, append exactly ONE marker:\n"
    "- [MORE] if there are more summaries still to produce on the next continuation\n"
    "- [DONE] if this is the complete, final batch\n"
    "The JSON array MUST contain complete, valid JSON objects. Never split an object across batches.\n\n"
    "Here are the current summaries:\n\n"
    "{summaries}"
)

_MERGE_CONTINUE = (
    "CONTINUATION — you are continuing to merge the same set of summaries.\n\n"
    "You already produced these merged summaries (do NOT repeat them):\n"
    "{produced}\n\n"
    "Continue merging the REMAINING summaries that are NOT already covered by the above.\n"
    "Produce the NEXT batch as a JSON array. Only include NEW merged summaries.\n"
    "If no remaining summaries need merging, output an empty array [] and [DONE].\n"
    "Otherwise output [MORE] if there are still more after this batch.\n\n"
    "Full context for reference:\n\n"
    "{summaries}"
)


def _parse_json_array(text: str) -> list | None:
    """Robustly parse a JSON array from LLM output. Handles code fences and markers."""
    if not text:
        return None
    sliced = text.strip()
    # Chop off trailing markers
    for marker in ["[MORE]", "[DONE]"]:
        if sliced.endswith(marker):
            sliced = sliced[:-len(marker)].rstrip()
    try:
        result = json.loads(sliced)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    m = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", sliced)
    if m:
        try:
            result = json.loads(m.group(1))
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
    start = sliced.find("[")
    end = sliced.rfind("]")
    if start != -1 and end > start:
        extracted = sliced[start:end + 1]
        for marker in ["[MORE]", "[DONE]"]:
            if extracted.endswith(marker):
                extracted = extracted[:-len(marker)].rstrip()
        try:
            result = json.loads(extracted)
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass
    return None


def _salvage_partial(text: str) -> list | None:
    """Recover complete JSON objects from a truncated array (LLM hit output limit).

    Walks forward extracting valid objects one at a time, then reassembles the array.
    """
    cleaned = text.strip()
    cleaned = re.sub(r'^```(?:json)?\s*\n?', '', cleaned)
    if not cleaned.startswith('['):
        return None

    inner = cleaned[1:].strip()
    results = []

    while inner:
        obj_start = inner.find('{')
        if obj_start == -1:
            break

        obj_str = None
        for end in range(obj_start + 1, len(inner) + 1):
            candidate = inner[obj_start:end]
            try:
                obj = json.loads(candidate)
                if isinstance(obj, dict):
                    obj_str = candidate
                    break
            except json.JSONDecodeError:
                continue

        if obj_str is None:
            break

        results.append(obj_str)
        consumed = obj_start + len(obj_str)
        inner = inner[consumed:].strip()
        if inner.startswith(','):
            inner = inner[1:].strip()

    if not results:
        return None

    try:
        arr = json.loads('[' + ','.join(results) + ']')
        if isinstance(arr, list):
            return arr
    except json.JSONDecodeError:
        pass
    return None


def _is_done(content: str, finish: str, chunk_num: int, total_merged: int) -> bool:
    if "[DONE]" in content:
        return True
    if chunk_num > 0 and content.strip() == "[]":
        return True
    if finish != "length" and not _has_marker(content, "[MORE]"):
        return True
    if len(content) < 7000 and not _has_marker(content, "[MORE]"):
        return True
    if total_merged >= MERGE_SUMMARIES_TARGET:
        return True
    return False


def _has_marker(text: str, marker: str) -> bool:
    return text.rstrip().endswith(marker)


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
