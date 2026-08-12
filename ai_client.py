"""LLM API clients — hosted OpenRouter brain + local vision model."""

import copy
import importlib
import json
import re
import sys
import threading
import time
import requests
from logger import info
from vision_history import VisionDescriptionLog, VisionRequestHistory
from config import (
    LLM_BASE_URL,
    LLM_API_KEY,
    LLM_MODEL,
    VISION_BASE_URL,
    VISION_MODEL,
    VISION_API_KEY,
    VISION_AARG_MLX_DIR,
    VISION_ENABLE_THINKING,
    VISION_MAX_TOKENS,
    VISION_PROMPT_BASE,
    VISION_PROVIDER,
    VISION_REQUESTS_FILE,
    VISION_THINKING_BUDGET,
    VISION_TIMEOUT,
    LLM_MAX_TOKENS,
    LLM_MAX_TOKENS_COMPACT,
    LLM_TIMEOUT,
    MERGE_SUMMARIES_TARGET,
    TOKEN_ESTIMATE_DIVISOR,
)


class LLMError(Exception):
    """LLM API error with HTTP status code for classification."""
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(message)


def _is_recoverable(status: int) -> bool:
    return status >= 500 or status == 429


def _api_call(url: str, headers: dict, payload: dict, timeout: int, caller: str = "") -> dict:
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if not resp.ok:
        body = resp.text[:500]
        status = resp.status_code
        info(f"[LLM] {caller}{status} error: {body}")
        raise LLMError(status, f"{status} HTTP error from LLM API")
    return resp.json()

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


# AARG's canonical perception schema sets additionalProperties=False, so without
# a field of its own everything update_vision_requests asks for is generated and
# then discarded. This is that field. It stays out of aarg_mlx/scene.py on
# purpose: the four canonical fields are enums built for frame-to-frame
# comparison (scene.diff_observations depends on that), while this one is
# free text the agent rewords constantly.
AARG_REQUEST_FIELD = "requested_observations"
AARG_REQUEST_STATUSES = ("present", "none", "unclear")
AARG_REQUEST_MAX_CHARS = 400

AARG_REQUEST_PREAMBLE = (
    "\n\nAdditional observation request. Answer it only in the "
    f"`{AARG_REQUEST_FIELD}` field; the four other fields keep the meaning and "
    'rules given above. Use status "present" when you can answer from this '
    'frame, "none" when the request does not apply to it, and "unclear" when '
    "the frame cannot support an answer. Report only what is visible, and keep "
    f"the description under {AARG_REQUEST_MAX_CHARS} characters.\n"
)


class AIClient:
    """Hosted brain LLM for reasoning and tool use."""

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

        data = _api_call(
            f"{self.base_url}/chat/completions",
            self._headers,
            payload,
            LLM_TIMEOUT,
            caller="chat_with_tools: ",
        )
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
        reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
        if not isinstance(reasoning, str):
            reasoning = str(reasoning)
        return {
            "content": content,
            "reasoning": reasoning.strip(),
            "reasoning_details": msg.get("reasoning_details") or [],
            "tool_calls": tool_calls,
            "raw_message": msg,
        }

    def compact(self, text: str) -> str:
        """Summarize old context using the configured brain model."""
        prompt = (
            f"{_PURPOSE}\n\n"
            "You are compacting a block of conversation history into a short summary for the AI's long-term memory.\n"
            "The input is a log of messages from the AI assistant and its user. Each line starts with a timestamp [Day HH:MM:SS] "
            "followed by the role (user/assistant/tool) and content.\n\n"
            "Write a dense, narrative summary (1-3 paragraphs) of what HAPPENED, not a line-by-line replay. "
            "Focus on the story, not the mechanics.\n\n"
            "PRIORITIZE (include these):\n"
            "- USER MESSAGES: every thing the user said — questions, preferences, corrections, personality, jokes\n"
            "- SCENE CHANGES: only mention when something DIFFERENT happened — user arrived, left, switched activities, "
            "changed lighting significantly. Do NOT repeat similar scene descriptions.\n"
            "- NOTIFICATIONS: what was proposed, approved/rejected, scheduled, or deleted\n"
            "- HABIT PATTERNS: e.g. user ignored stretch reminders, worked past midnight, actually got up and moved\n"
            "- CONVERSATION TOPICS: what was discussed, running jokes, things the user expressed interest in\n"
            "- INTERESTING SEARCH RESULTS: key facts or finds the AI shared\n"
            "- USER ENGAGEMENT: whether the user was chatting, pressing buttons, or absent\n\n"
            "DROP / CONDENSE (skip or mention once):\n"
            "- Routine photo descriptions where nothing changed (same person at desk, same lighting)\n"
            "- Wait cycles with no interruptions\n"
            "- Boilerplate update_display / send_chat_message results (their content matters, the tool call doesn't)\n"
            "- Redundant scene descriptions — if the scene barely changed, don't describe it again\n"
            "- Restarts and boot messages — mention once if it happened multiple times\n\n"
            "FORMAT: Write a plain narrative paragraph. No bullet points, no markdown, no timestamps in the output. "
            "Just tell what happened during this time period.\n\n"
            f"Here is the log to summarize:\n\n{text}"
        )
        messages = [
            {"role": "user", "content": prompt}
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": LLM_MAX_TOKENS_COMPACT,
            "temperature": 0.3,
        }
        resp = _api_call(
            f"{self.base_url}/chat/completions",
            self._headers,
            payload,
            LLM_TIMEOUT,
            caller="compact: ",
        )
        return resp["choices"][0]["message"]["content"].strip()

    def merge_summaries(self, summaries_text: str) -> list:
        """Merge summaries using the configured brain model."""
        info(f"[LLM] merge_summaries: {len(summaries_text)} chars input, targeting <= {MERGE_SUMMARIES_TARGET} summaries")
        prompt = _MERGE_PROMPT.format(
            target=MERGE_SUMMARIES_TARGET,
            purpose=_PURPOSE,
            summaries=summaries_text,
        )
        content, finish = self._merge_call(prompt)
        info(f"[LLM] merge_summaries: {len(content)} chars, finish={finish}")
        if not content:
            return []
        result = _parse_json_array(content)
        if result is None:
            result = _salvage_partial(content)
            if result is not None:
                info(f"[LLM] merge_summaries: salvaged {len(result)} objects from truncated response")
        if result:
            info(f"[LLM] merge_summaries: {len(result)} summaries")
        else:
            info(f"[LLM] merge_summaries: unparseable, raw={content[:200]}")
        return result or []

    def _merge_call(self, prompt: str) -> tuple:
        """Make a merge/completion call. Returns (content, finish_reason)."""
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 40960,
            "temperature": 0.3,
        }
        resp = _api_call(
            f"{self.base_url}/chat/completions",
            self._headers,
            payload,
            300,
            caller="_merge_call: ",
        )
        if "choices" not in resp:
            info(f"[LLM] _merge_call: no 'choices' in response. Keys: {list(resp.keys())}. Body: {json.dumps(resp)[:300]}")
            return "", "error"
        choice = resp["choices"][0]
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
    "- MERGE summaries that are adjacent in time into one — only combine consecutive summary numbers (a gap is fine only if you are dropping the summaries in the gap). Never merge periods from far apart in time.\n"
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
    "Each input summary below is numbered like [1], [2], etc.\n\n"
    "Return a JSON array of objects. Each object has:\n"
    '- "sources": array of input summary numbers this entry covers, e.g. [3, 4, 5] (dropped summaries simply appear in no entry)\n'
    '- "time_range": the time period covered\n'
    '- "summary": the summary text\n\n'
    "TARGET: produce at most {target} summaries.\n"
    "Return ONLY the JSON array, no other text.\n\n"
    "Here are the current summaries:\n\n"
    "{summaries}"
)


def _parse_json_array(text: str) -> list | None:
    """Robustly parse a JSON array from LLM output. Handles code fences."""
    if not text:
        return None
    sliced = text.strip()
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
        try:
            result = json.loads(sliced[start:end + 1])
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


class VisionClient:
    """Local vision client with generic llama.cpp and AARG MLX modes."""

    def __init__(
        self,
        request_history: VisionRequestHistory | None = None,
        description_log: VisionDescriptionLog | None = None,
    ):
        self.base_url = VISION_BASE_URL.rstrip("/")
        self.model = VISION_MODEL
        self.provider = VISION_PROVIDER
        self.request_history = request_history
        self.description_log = description_log
        self.last_request_commit = ""
        # Mode the stored requests were written for, when it isn't this
        # provider's. Empty means they applied normally.
        self.requests_stale = ""
        self._inference_lock = threading.Lock()
        self._aarg_scene = None
        self._headers = {}
        if VISION_API_KEY:
            self._headers["Authorization"] = f"Bearer {VISION_API_KEY}"
        if self.provider == "aarg_mlx":
            self._aarg_scene = self._load_aarg_scene()
        elif self.provider != "generic":
            raise ValueError("VISION_PROVIDER must be 'generic' or 'aarg_mlx'")

    @staticmethod
    def _load_aarg_scene():
        """Load the canonical AARG prompt/schema client only in AARG mode."""
        if VISION_AARG_MLX_DIR and VISION_AARG_MLX_DIR not in sys.path:
            sys.path.insert(0, VISION_AARG_MLX_DIR)
        try:
            module = importlib.import_module("scene")
        except ImportError as exc:
            raise RuntimeError(
                "VISION_PROVIDER=aarg_mlx requires scene.py from aarg_mlx; set "
                "VISION_AARG_MLX_DIR or add that directory to PYTHONPATH"
            ) from exc
        required = ("build_perception_payload", "validate_perception")
        if not all(hasattr(module, name) for name in required):
            location = getattr(module, "__file__", "unknown location")
            raise RuntimeError(f"Imported scene module at {location} is not the AARG client")
        return module

    def _build_vision_prompt(self, requests_content: str | None = None) -> str:
        """Build vision prompt from base + requests file."""
        prompt = VISION_PROMPT_BASE
        if requests_content is None:
            try:
                with open(VISION_REQUESTS_FILE, "r") as f:
                    requests_content = f.read()
            except FileNotFoundError:
                requests_content = ""
        extra = requests_content.strip()
        if extra:
            prompt += "\n\n" + extra
        return prompt

    def _build_generic_payload(
        self, image_data_uri: str, requests_content: str | None = None
    ) -> dict:
        return {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self._build_vision_prompt(requests_content)},
                        {"type": "image_url", "image_url": {"url": image_data_uri}},
                    ],
                }
            ],
            "max_tokens": VISION_MAX_TOKENS,
            "temperature": 0.3,
        }

    def _build_aarg_payload(
        self, image_data_uri: str, requests_content: str | None = None
    ) -> dict:
        """Use AARG's canonical image ordering, prompt, and strict schema."""
        payload = self._aarg_scene.build_perception_payload(
            model=self.model,
            image_data_uri=image_data_uri,
            max_tokens=VISION_MAX_TOKENS,
        )

        # Preserve AI Roommate's update_vision_requests tool while retaining the
        # canonical AARG base prompt and schema.
        if requests_content is None:
            try:
                with open(VISION_REQUESTS_FILE, "r") as f:
                    requests_content = f.read()
            except FileNotFoundError:
                requests_content = ""
        extra_prompt = requests_content.strip()
        answer_tokens = VISION_MAX_TOKENS
        if extra_prompt:
            content = payload["messages"][0]["content"]
            for part in reversed(content):
                if part.get("type") == "text":
                    part["text"] += AARG_REQUEST_PREAMBLE + extra_prompt
                    break
            if self._add_requested_observations(payload):
                # The extra field has to fit alongside the canonical answer or
                # the whole response truncates into unparseable JSON.
                answer_tokens += AARG_REQUEST_MAX_CHARS // 3
        payload["max_tokens"] = answer_tokens

        if VISION_ENABLE_THINKING:
            payload.update({
                "enable_thinking": True,
                "thinking_budget": VISION_THINKING_BUDGET,
                "thinking_start_token": "<|think|>",
                "thinking_end_token": "<channel|>",
                # Reserve the normal structured answer budget after thinking.
                "max_tokens": VISION_THINKING_BUDGET + answer_tokens,
            })
        return payload

    @staticmethod
    def _add_requested_observations(payload: dict) -> bool:
        """Add the custom-request field to this payload's copy of the schema.

        build_perception_payload() hands back the module-level PERCEPTION_SCHEMA
        by reference, so the deep copy is required, not hygiene: editing in place
        would change the schema for every other importer of scene.py in the
        process, including its CLI and eval paths.
        """
        try:
            json_schema = payload["response_format"]["json_schema"]
            schema = copy.deepcopy(json_schema["schema"])
            properties = schema["properties"]
        except (KeyError, TypeError):
            info("[VISION] AARG payload exposes no schema; sending request as text only")
            return False

        properties[AARG_REQUEST_FIELD] = {
            "type": "object",
            "additionalProperties": False,
            "required": ["status", "description"],
            "properties": {
                "status": {"type": "string", "enum": list(AARG_REQUEST_STATUSES)},
                "description": {"type": "string", "maxLength": AARG_REQUEST_MAX_CHARS},
            },
        }
        required = schema.setdefault("required", [])
        if AARG_REQUEST_FIELD not in required:
            required.append(AARG_REQUEST_FIELD)
        json_schema["schema"] = schema
        return True

    @staticmethod
    def _message_text(data: dict) -> str:
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError(f"Unexpected vision response: {data}") from exc
        if isinstance(content, list):
            content = "".join(
                item.get("text", "")
                for item in content
                if isinstance(item, dict)
                and item.get("type") in ("text", "output_text")
            )
        if not isinstance(content, str):
            raise ValueError(f"Vision response content is not text: {content!r}")
        return content.strip()

    def _parse_aarg_perception(self, data: dict) -> dict:
        """Parse the final structured result, tolerating exposed thought text."""
        text = self._message_text(data)
        if "<channel|>" in text:
            text = text.rsplit("<channel|>", 1)[-1].strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        candidates = [text]
        candidates.extend(text[index:] for index, char in enumerate(text) if char == "{")
        decoder = json.JSONDecoder()
        last_error = None
        for candidate in candidates:
            try:
                perception, _ = decoder.raw_decode(candidate.lstrip())
                if not isinstance(perception, dict):
                    continue
                # Tolerates extra keys: it only checks for missing ones.
                self._aarg_scene.validate_perception(perception)
                self._check_requested_observations(perception)
                return perception
            except (json.JSONDecodeError, ValueError, TypeError, RuntimeError) as exc:
                last_error = exc
        raise ValueError(f"AARG vision returned no valid perception: {text[:500]}") from last_error

    @staticmethod
    def _check_requested_observations(perception: dict):
        """Validate the one field scene.validate_perception doesn't know about.

        Drops it instead of raising. A malformed custom field should not cost the
        frame its four canonical fields, which is what a raise here would do:
        _parse_aarg_perception would move on to the next candidate and most
        likely fail the whole description.
        """
        value = perception.get(AARG_REQUEST_FIELD)
        if value is None:
            return
        valid = (
            isinstance(value, dict)
            and value.get("status") in AARG_REQUEST_STATUSES
            and isinstance(value.get("description"), str)
        )
        if not valid:
            info(f"[VISION] Dropping malformed {AARG_REQUEST_FIELD}: {value!r}")
            perception.pop(AARG_REQUEST_FIELD, None)

    def health_check(self) -> bool:
        """Return whether the configured vision service is accepting requests."""
        try:
            response = requests.get(f"{self.base_url}/models", timeout=5)
            return response.ok
        except requests.RequestException:
            return False

    def _record_description(
        self,
        description: str,
        *,
        request_commit: str,
        source: str,
        captured_at: float | None,
        latency_s: float,
        usage=None,
        timings=None,
    ):
        self.last_request_commit = request_commit
        if not self.description_log:
            return
        try:
            self.description_log.append(
                description=description,
                request_commit=request_commit,
                model=self.model,
                provider=self.provider,
                source=source,
                captured_at=captured_at,
                latency_s=latency_s,
                usage=usage,
                timings=timings,
            )
        except Exception as exc:
            info(f"[VISION] Description log error: {exc}")

    def describe(
        self,
        image_data_uri: str,
        max_retries: int = 3,
        *,
        source: str = "camera",
        captured_at: float | None = None,
    ) -> str:
        """Describe a frame, allowing only one active local inference request."""
        requests_content = None
        request_commit = ""
        if self.request_history:
            requests_content, request_commit, declared = (
                self.request_history.snapshot_for(self.provider)
            )
            if requests_content:
                self.requests_stale = ""
            else:
                self.requests_stale = declared if declared != self.provider else ""
                if self.requests_stale:
                    info(
                        f"[VISION] Requests declare {declared}, provider is "
                        f"{self.provider}; not appending them"
                    )
        payload = (
            self._build_aarg_payload(image_data_uri, requests_content)
            if self.provider == "aarg_mlx"
            else self._build_generic_payload(image_data_uri, requests_content)
        )
        with self._inference_lock:
            for attempt in range(max_retries):
                started = time.perf_counter()
                try:
                    resp = requests.post(
                        f"{self.base_url}/chat/completions",
                        headers=self._headers,
                        json=payload,
                        timeout=VISION_TIMEOUT,
                    )
                    status = getattr(resp, "status_code", 200)
                    if (
                        self.provider == "aarg_mlx"
                        and (status >= 500 or status == 429)
                        and attempt + 1 < max_retries
                    ):
                        delay = 2 ** attempt
                        info(f"[VISION] AARG HTTP {status}; retrying in {delay}s")
                        time.sleep(delay)
                        continue
                    resp.raise_for_status()
                except (requests.ConnectionError, requests.Timeout) as exc:
                    if attempt + 1 >= max_retries:
                        raise
                    delay = 2 ** attempt
                    info(f"[VISION] Transient request failure: {exc}; retrying in {delay}s")
                    time.sleep(delay)
                    continue

                data = resp.json()
                elapsed = time.perf_counter() - started
                if self.provider == "aarg_mlx":
                    perception = self._parse_aarg_perception(data)
                    usage = data.get("usage") or {}
                    timings = data.get("timings") or {}
                    info(
                        f"[VISION] AARG perception in {elapsed:.2f}s; "
                        f"usage={usage}, timings={timings}, thinking={VISION_ENABLE_THINKING}"
                    )
                    description = json.dumps(
                        perception, ensure_ascii=False, separators=(",", ":")
                    )
                    self._record_description(
                        description,
                        request_commit=request_commit,
                        source=source,
                        captured_at=captured_at,
                        latency_s=elapsed,
                        usage=usage,
                        timings=timings,
                    )
                    return description

                choice = data["choices"][0]
                finish = choice.get("finish_reason", "unknown")
                raw = self._message_text(data)
                cleaned = _clean(raw)
                if cleaned:
                    if finish == "length":
                        info(f"[VISION] Description truncated (hit max_tokens). Length: {len(cleaned)} chars")
                    self._record_description(
                        cleaned,
                        request_commit=request_commit,
                        source=source,
                        captured_at=captured_at,
                        latency_s=elapsed,
                        usage=data.get("usage"),
                        timings=data.get("timings"),
                    )
                    return cleaned
                info(f"[VISION] Empty response (attempt {attempt + 1}/{max_retries}). Raw: {repr(raw[:200])}")
        return ""
