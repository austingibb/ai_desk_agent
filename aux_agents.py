"""Config-driven auxiliary LLM agents for mechanical tool delegation."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from urllib.parse import urlparse

import requests

from logger import info


SCHEMA_VERSION = 1
CLASSIFIER_TEMPERATURE = 0.6
CLASSIFIER_TOP_P = 0.95
CLASSIFIER_MAX_TOKENS = 200
WORK_MAX_TOKENS = 20_000
WORK_TIMEOUT_SECONDS = 120
BACKOFF_BASE_SECONDS = 5
BACKOFF_MAX_SECONDS = 300
PRIVATE_HISTORY_RECORDS = 24
CLASSIFIER_CONTEXT_MESSAGES = 6
WORK_CONTEXT_MESSAGES = 12
CLASSIFIER_CONTEXT_CHARS = 6_000
WORK_CONTEXT_CHARS = 16_000


class AuxiliaryConfigError(ValueError):
    """Invalid auxiliary-agent configuration."""


class AuxiliaryProtocolError(RuntimeError):
    """An auxiliary endpoint returned an unusable response."""


@dataclass(frozen=True)
class AuxiliaryAgentConfig:
    name: str
    base_url: str
    model: str
    tools: tuple[str, ...]
    classification_timeout_seconds: float = 2.0
    api_key_env: str = ""
    api_key: str = field(default="", repr=False)


@dataclass
class _AgentState:
    config: AuxiliaryAgentConfig
    owned_tools: tuple[str, ...]
    history: deque = field(
        default_factory=lambda: deque(maxlen=PRIVATE_HISTORY_RECORDS)
    )
    failure_count: int = 0
    next_retry_at: float = 0.0
    outage_warned: bool = False
    last_revision: str = ""
    prior_entry_hashes: tuple[str, ...] = ()


@dataclass
class AuxiliaryTurn:
    response: dict | None
    agent_name: str | None
    context_revision: str
    escalation_packet: dict | None


class AuxiliaryAuditLog:
    """Append-only, local JSONL record of auxiliary activity."""

    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        self._lock = threading.Lock()
        self._write_warned = False

    def append(self, event: dict) -> dict:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **copy.deepcopy(event),
        }
        with self._lock:
            try:
                os.makedirs(os.path.dirname(self.path), exist_ok=True)
                with open(self.path, "a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(record, ensure_ascii=False, default=str) + "\n"
                    )
                self._write_warned = False
            except OSError as exc:
                if not self._write_warned:
                    info(f"[AUX WARNING] Could not write audit log {self.path}: {exc}")
                    self._write_warned = True
        return record


def _config_error(message: str, valid_tool_names: set[str]) -> AuxiliaryConfigError:
    valid = ", ".join(sorted(valid_tool_names)) or "(none enabled)"
    return AuxiliaryConfigError(f"{message}. Valid tool names: {valid}")


def _require_nonempty_string(value, field_name: str, valid_tools: set[str]) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _config_error(f"{field_name} must be a non-empty string", valid_tools)
    return value.strip()


def _validate_base_url(value, valid_tools: set[str]) -> str:
    base_url = _require_nonempty_string(value, "base_url", valid_tools).rstrip("/")
    parsed = urlparse(base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise _config_error("base_url must be an absolute HTTP(S) URL", valid_tools)
    if parsed.query or parsed.fragment:
        raise _config_error("base_url cannot contain a query or fragment", valid_tools)
    return base_url


def _parse_config(
    data,
    *,
    active_tool_names: set[str],
    registered_tool_names: set[str],
    forbidden_tool_names: set[str],
) -> list[AuxiliaryAgentConfig]:
    if not isinstance(data, dict):
        raise _config_error("auxiliary config root must be an object", active_tool_names)
    extra_root = set(data).difference({"schema_version", "agents"})
    if extra_root:
        raise _config_error(
            f"unknown root field(s): {', '.join(sorted(extra_root))}",
            active_tool_names,
        )
    schema_version = data.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != SCHEMA_VERSION
    ):
        raise _config_error(
            f"unsupported schema_version {schema_version!r}; expected {SCHEMA_VERSION}",
            active_tool_names,
        )
    agents = data.get("agents")
    if not isinstance(agents, list):
        raise _config_error("agents must be an array", active_tool_names)

    allowed_fields = {
        "name",
        "base_url",
        "model",
        "tools",
        "classification_timeout_seconds",
        "api_key_env",
    }
    known_tools = set(registered_tool_names) | set(active_tool_names)
    parsed_agents = []
    seen_names = set()

    for index, raw in enumerate(agents):
        prefix = f"agents[{index}]"
        if not isinstance(raw, dict):
            raise _config_error(f"{prefix} must be an object", active_tool_names)
        extra_fields = set(raw).difference(allowed_fields)
        if extra_fields:
            raise _config_error(
                f"{prefix} has unknown field(s): {', '.join(sorted(extra_fields))}",
                active_tool_names,
            )

        name = _require_nonempty_string(raw.get("name"), f"{prefix}.name", active_tool_names)
        if name in seen_names:
            raise _config_error(f"duplicate agent name {name!r}", active_tool_names)
        seen_names.add(name)
        base_url = _validate_base_url(raw.get("base_url"), active_tool_names)
        model = _require_nonempty_string(raw.get("model"), f"{prefix}.model", active_tool_names)

        configured_tools = raw.get("tools")
        if not isinstance(configured_tools, list) or not configured_tools:
            raise _config_error(
                f"{prefix}.tools must be a non-empty array", active_tool_names
            )
        if not all(isinstance(tool, str) and tool for tool in configured_tools):
            raise _config_error(
                f"{prefix}.tools entries must be non-empty strings", active_tool_names
            )
        if len(set(configured_tools)) != len(configured_tools):
            raise _config_error(f"{prefix}.tools contains duplicates", active_tool_names)

        unknown = set(configured_tools).difference(known_tools)
        if unknown:
            raise _config_error(
                f"{prefix}.tools contains unknown tool(s): {', '.join(sorted(unknown))}",
                active_tool_names,
            )
        prohibited = set(configured_tools).intersection(forbidden_tool_names)
        if prohibited:
            raise _config_error(
                f"{prefix}.tools contains non-delegatable tool(s): "
                f"{', '.join(sorted(prohibited))}",
                active_tool_names,
            )

        timeout = raw.get("classification_timeout_seconds", 2.0)
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise _config_error(
                f"{prefix}.classification_timeout_seconds must be a positive finite number",
                active_tool_names,
            )

        api_key_env = raw.get("api_key_env", "")
        if api_key_env:
            api_key_env = _require_nonempty_string(
                api_key_env, f"{prefix}.api_key_env", active_tool_names
            )
            api_key = os.environ.get(api_key_env, "")
            if not api_key:
                raise _config_error(
                    f"{prefix}.api_key_env names unset environment variable {api_key_env!r}",
                    active_tool_names,
                )
        else:
            api_key_env = ""
            api_key = ""

        active_configured_tools = tuple(
            tool for tool in configured_tools if tool in active_tool_names
        )
        parsed_agents.append(
            AuxiliaryAgentConfig(
                name=name,
                base_url=base_url,
                model=model,
                tools=active_configured_tools,
                classification_timeout_seconds=float(timeout),
                api_key_env=api_key_env,
                api_key=api_key,
            )
        )
    return parsed_agents


def load_auxiliary_manager(
    *,
    enabled: bool,
    config_path: str,
    tool_definitions: list[dict],
    registered_tool_names: set[str],
    forbidden_tool_names: set[str],
    log_path: str,
    max_tool_calls_per_turn: int,
    post=None,
    monotonic=None,
) -> "AuxiliaryAgentManager | None":
    """Load and validate the optional config, returning None for no-op modes."""
    if not enabled or not os.path.exists(config_path):
        return None

    active_tools = {
        tool["function"]["name"]: copy.deepcopy(tool)
        for tool in tool_definitions
    }
    active_names = set(active_tools)
    try:
        with open(config_path, encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise _config_error(
            f"malformed auxiliary config JSON at line {exc.lineno}, column {exc.colno}",
            active_names,
        ) from exc
    except OSError as exc:
        raise _config_error(
            f"could not read auxiliary config {config_path!r}: {exc}", active_names
        ) from exc

    configs = _parse_config(
        data,
        active_tool_names=active_names,
        registered_tool_names=set(registered_tool_names),
        forbidden_tool_names=set(forbidden_tool_names),
    )
    if not configs:
        return None

    claimed = set()
    states = []
    for config in configs:
        owned = tuple(tool for tool in config.tools if tool not in claimed)
        claimed.update(owned)
        if owned:
            states.append(_AgentState(config=config, owned_tools=owned))
        else:
            info(f"[AUX] {config.name} has no active unclaimed tools; skipping")
    if not states:
        return None

    return AuxiliaryAgentManager(
        states=states,
        tool_definitions=active_tools,
        audit_log=AuxiliaryAuditLog(log_path),
        max_tool_calls_per_turn=max_tool_calls_per_turn,
        post=post,
        monotonic=monotonic,
    )


def _text_content(content) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else str(content)
    text_parts = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") in {"text", "input_text", "output_text"}:
            text = part.get("text", "")
            if isinstance(text, str) and text:
                text_parts.append(text)
    return "\n".join(text_parts)


def _context_entries(messages: list[dict]) -> list[dict]:
    entries = []
    for message in messages:
        role = message.get("role", "")
        if role == "system":
            continue
        entry = {"role": role, "content": _text_content(message.get("content"))}
        if message.get("name"):
            entry["name"] = str(message["name"])
        tool_calls = []
        for tool_call in message.get("tool_calls", []):
            function = tool_call.get("function", {})
            tool_calls.append({
                "name": function.get("name", ""),
                "arguments": function.get("arguments", "{}"),
            })
        if tool_calls:
            entry["tool_calls"] = tool_calls
        entries.append(entry)
    return entries


def _entry_hashes(entries: list[dict]) -> tuple[str, ...]:
    return tuple(
        hashlib.sha256(
            json.dumps(entry, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        for entry in entries
    )


def _context_revision(entries: list[dict]) -> str:
    digest = hashlib.sha256()
    for entry_hash in _entry_hashes(entries):
        digest.update(entry_hash.encode("ascii"))
    return digest.hexdigest()[:20]


def _bounded_entries(entries: list[dict], max_messages: int, max_chars: int) -> list[dict]:
    selected = []
    used = 0
    for entry in reversed(entries[-max_messages:]):
        size = len(json.dumps(entry, ensure_ascii=False, default=str))
        if selected and used + size > max_chars:
            break
        if size > max_chars:
            clipped = copy.deepcopy(entry)
            clipped["content"] = clipped.get("content", "")[-max_chars:]
            selected.append(clipped)
            break
        selected.append(copy.deepcopy(entry))
        used += size
    selected.reverse()
    return selected


def _context_delta(state: _AgentState, entries: list[dict]) -> list[dict]:
    current = _entry_hashes(entries)
    previous = state.prior_entry_hashes
    common = 0
    while common < min(len(current), len(previous)) and current[common] == previous[common]:
        common += 1
    delta = entries[common:] if common else entries[-CLASSIFIER_CONTEXT_MESSAGES:]
    return _bounded_entries(
        delta,
        CLASSIFIER_CONTEXT_MESSAGES,
        CLASSIFIER_CONTEXT_CHARS,
    )


def _message_content(message: dict) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return _text_content(content).strip()
    return str(content).strip() if content is not None else ""


def _parse_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    decoder = json.JSONDecoder()
    candidates = [text]
    candidates.extend(text[index:] for index, char in enumerate(text) if char == "{")
    for candidate in candidates:
        try:
            value, _ = decoder.raw_decode(candidate.lstrip())
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise AuxiliaryProtocolError("response contained no JSON object")


def _normalize_tool_response(data: dict) -> tuple[dict, dict]:
    try:
        message = data["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise AuxiliaryProtocolError("response has no choices[0].message") from exc
    if not isinstance(message, dict):
        raise AuxiliaryProtocolError("response message is not an object")

    tool_calls = []
    for index, raw_call in enumerate(message.get("tool_calls") or []):
        try:
            function = raw_call["function"]
            name = function["name"]
            arguments = function.get("arguments", {})
        except (KeyError, TypeError) as exc:
            raise AuxiliaryProtocolError("malformed tool call") from exc
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as exc:
                raise AuxiliaryProtocolError(
                    f"tool {name!r} returned invalid JSON arguments"
                ) from exc
        if not isinstance(arguments, dict):
            raise AuxiliaryProtocolError(
                f"tool {name!r} arguments must be an object"
            )
        tool_calls.append({
            # Provider IDs are not guaranteed unique across independent local
            # requests. Canonical context requires globally distinct pairing IDs.
            "id": f"aux_{uuid.uuid4().hex}_{index}",
            "name": str(name),
            "arguments": arguments,
        })

    normalized = {
        # Auxiliary prose is deliberately discarded; only tool calls can enter
        # canonical context or become visible to the user.
        "content": "",
        "reasoning": "",
        "reasoning_details": [],
        "tool_calls": tool_calls,
        "raw_message": message,
    }
    return normalized, message


class AuxiliaryAgentManager:
    def __init__(
        self,
        *,
        states: list[_AgentState],
        tool_definitions: dict[str, dict],
        audit_log: AuxiliaryAuditLog,
        max_tool_calls_per_turn: int,
        post=None,
        monotonic=None,
    ):
        self._states = states
        self._states_by_name = {state.config.name: state for state in states}
        self._tool_definitions = tool_definitions
        self.audit_log = audit_log
        self._max_tool_calls_per_turn = max_tool_calls_per_turn
        self._post = post or requests.post
        self._monotonic = monotonic or time.monotonic

    @property
    def agents(self) -> tuple[AuxiliaryAgentConfig, ...]:
        return tuple(
            replace(state.config, tools=state.owned_tools)
            for state in self._states
        )

    def owner_for(self, tool_name: str) -> str | None:
        for state in self._states:
            if tool_name in state.owned_tools:
                return state.config.name
        return None

    @staticmethod
    def _headers(config: AuxiliaryAgentConfig) -> dict:
        headers = {"Content-Type": "application/json"}
        if config.api_key:
            headers["Authorization"] = f"Bearer {config.api_key}"
        return headers

    @staticmethod
    def _logged_headers(headers: dict) -> dict:
        return {
            key: "[REDACTED]" if key.lower() == "authorization" else value
            for key, value in headers.items()
        }

    def _request(
        self,
        state: _AgentState,
        *,
        stage: str,
        payload: dict,
        timeout: float,
        revision: str,
    ) -> tuple[dict, float, dict]:
        url = f"{state.config.base_url}/chat/completions"
        headers = self._headers(state.config)
        request_record = {
            "url": url,
            "headers": self._logged_headers(headers),
            "payload": copy.deepcopy(payload),
            "timeout_seconds": timeout,
        }
        started = self._monotonic()
        response_record = None
        try:
            response = self._post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            response_record = {
                "status_code": getattr(response, "status_code", None),
                "text": getattr(response, "text", ""),
            }
            response.raise_for_status()
            data = response.json()
            response_record["json"] = copy.deepcopy(data)
            return data, self._monotonic() - started, request_record
        except Exception as exc:
            elapsed = self._monotonic() - started
            setattr(exc, "_aux_request_record", request_record)
            setattr(exc, "_aux_response_record", response_record)
            setattr(exc, "_aux_latency_s", elapsed)
            setattr(exc, "_aux_stage", stage)
            setattr(exc, "_aux_revision", revision)
            raise

    def _classification_system_prompt(self, state: _AgentState) -> str:
        capabilities = []
        for name in state.owned_tools:
            description = str(
                self._tool_definitions[name]["function"].get("description", "")
            ).replace("\n", " ").strip()
            capabilities.append(f"- {name}: {description[:240]}")
        return (
            "You are a fast routing gate for a mechanical auxiliary agent named "
            f"{state.config.name}. Think briefly and do not solve the event. Decide "
            "whether the newest context clearly warrants a substantive turn by this "
            "agent using one of its capabilities. User-facing replies, ambiguity, and "
            "general judgment belong to the hosted brain. Return only the required "
            "boolean JSON.\n\nCapabilities:\n"
            + "\n".join(capabilities)
        )

    def _classifier_payload(self, state: _AgentState, delta: list[dict]) -> dict:
        return {
            "model": state.config.model,
            "messages": [
                {"role": "system", "content": self._classification_system_prompt(state)},
                {
                    "role": "user",
                    "content": (
                        "Classify only this new context delta:\n"
                        + json.dumps(delta, ensure_ascii=False, default=str)
                    ),
                },
            ],
            "temperature": CLASSIFIER_TEMPERATURE,
            "top_p": CLASSIFIER_TOP_P,
            "reasoning_effort": "low",
            "max_tokens": CLASSIFIER_MAX_TOKENS,
            "cache_prompt": True,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "delegation_decision",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {"delegate": {"type": "boolean"}},
                        "required": ["delegate"],
                    },
                },
            },
        }

    def _private_history(self, state: _AgentState) -> list[dict]:
        selected = []
        used = 0
        for record in reversed(state.history):
            size = len(json.dumps(record, ensure_ascii=False, default=str))
            if selected and used + size > WORK_CONTEXT_CHARS:
                break
            selected.append(copy.deepcopy(record))
            used += size
        selected.reverse()
        return selected

    def _work_payload(self, state: _AgentState, recent_context: list[dict]) -> dict:
        tools = [copy.deepcopy(self._tool_definitions[name]) for name in state.owned_tools]
        system = (
            "You are a private mechanical auxiliary agent. You have no user-facing "
            "voice. Use only the provided tools, and call them only when the context "
            "makes the action and arguments clear. Never invent missing facts, repeat "
            "a completed action, or answer the user. If judgment or a user-facing "
            "response is needed, make no tool call so the hosted brain can handle it."
        )
        content = {
            "recent_canonical_context": recent_context,
            "private_recent_history": self._private_history(state),
        }
        return {
            "model": state.config.model,
            "messages": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": json.dumps(content, ensure_ascii=False, default=str),
                },
            ],
            "max_tokens": WORK_MAX_TOKENS,
            "cache_prompt": True,
            "tools": tools,
            "tool_choice": "auto",
        }

    def _record_recovery(self, state: _AgentState, stage: str, revision: str):
        if not state.failure_count:
            return
        previous_failures = state.failure_count
        state.failure_count = 0
        state.next_retry_at = 0.0
        state.outage_warned = False
        self.audit_log.append({
            "event": "recovery",
            "agent": state.config.name,
            "model": state.config.model,
            "stage": stage,
            "context_revision": revision,
            "previous_failure_count": previous_failures,
        })

    def _record_failure(
        self,
        state: _AgentState,
        error: Exception,
        *,
        stage: str,
        revision: str,
    ) -> dict:
        state.failure_count += 1
        delay = min(
            BACKOFF_BASE_SECONDS * (2 ** (state.failure_count - 1)),
            BACKOFF_MAX_SECONDS,
        )
        state.next_retry_at = self._monotonic() + delay
        warned = not state.outage_warned
        if warned:
            info(
                f"[AUX WARNING] {state.config.name} unavailable during {stage}: "
                f"{error}; falling back to hosted brain"
            )
            state.outage_warned = True
        self.audit_log.append({
            "event": "failure",
            "agent": state.config.name,
            "base_url": state.config.base_url,
            "model": state.config.model,
            "stage": stage,
            "context_revision": revision,
            "error_type": type(error).__name__,
            "error": str(error),
            "latency_ms": round(getattr(error, "_aux_latency_s", 0.0) * 1000, 3),
            "request": getattr(error, "_aux_request_record", None),
            "response": getattr(error, "_aux_response_record", None),
            "failure_count": state.failure_count,
            "retry_in_seconds": delay,
            "operator_warning_emitted": warned,
        })
        return {
            "agent": state.config.name,
            "outcome": "failure",
            "stage": stage,
            "reason": str(error),
            "retry_in_seconds": delay,
        }

    def _validate_calls(self, state: _AgentState, response: dict):
        tool_calls = response.get("tool_calls", [])
        if len(tool_calls) > self._max_tool_calls_per_turn:
            raise AuxiliaryProtocolError(
                f"too many tool calls: {len(tool_calls)}; max is "
                f"{self._max_tool_calls_per_turn}"
            )
        unauthorized = [
            call["name"]
            for call in tool_calls
            if call["name"] not in state.owned_tools
        ]
        if unauthorized:
            raise AuxiliaryProtocolError(
                "unauthorized tool call(s): " + ", ".join(unauthorized)
            )

    def evaluate(self, messages: list[dict], tools: list[dict]) -> AuxiliaryTurn:
        """Try configured agents in order, returning a validated tool response or packet."""
        del tools  # Active definitions were frozen after startup validation.
        entries = _context_entries(messages)
        revision = _context_revision(entries)
        recent_context = _bounded_entries(
            entries, WORK_CONTEXT_MESSAGES, WORK_CONTEXT_CHARS
        )
        attempts = []

        for state in self._states:
            now = self._monotonic()
            if now < state.next_retry_at:
                remaining = max(0.0, state.next_retry_at - now)
                attempt = {
                    "agent": state.config.name,
                    "outcome": "backoff",
                    "retry_in_seconds": round(remaining, 3),
                }
                attempts.append(attempt)
                self.audit_log.append({
                    "event": "backoff_skip",
                    "model": state.config.model,
                    "context_revision": revision,
                    **attempt,
                })
                continue
            if state.last_revision == revision:
                attempts.append({
                    "agent": state.config.name,
                    "outcome": "already_classified",
                })
                continue

            delta = _context_delta(state, entries)
            classifier_payload = self._classifier_payload(state, delta)
            data = None
            latency = 0.0
            request_record = None
            try:
                data, latency, request_record = self._request(
                    state,
                    stage="classification",
                    payload=classifier_payload,
                    timeout=state.config.classification_timeout_seconds,
                    revision=revision,
                )
                normalized, raw_message = _normalize_tool_response(data)
                if normalized["tool_calls"]:
                    raise AuxiliaryProtocolError(
                        "classifier returned a tool call instead of boolean JSON"
                    )
                decision = _parse_json_object(_message_content(raw_message))
                if not isinstance(decision.get("delegate"), bool):
                    raise AuxiliaryProtocolError(
                        "classifier JSON must contain boolean delegate"
                    )
                self._record_recovery(state, "classification", revision)
            except Exception as error:
                if request_record is not None:
                    setattr(error, "_aux_request_record", request_record)
                    setattr(error, "_aux_response_record", {"json": copy.deepcopy(data)})
                    setattr(error, "_aux_latency_s", latency)
                attempts.append(
                    self._record_failure(
                        state,
                        error,
                        stage="classification",
                        revision=revision,
                    )
                )
                continue

            state.history.append({
                "type": "classification",
                "context_revision": revision,
                "delegate": decision["delegate"],
            })
            classification_event = {
                "event": "classification",
                "agent": state.config.name,
                "base_url": state.config.base_url,
                "model": state.config.model,
                "context_revision": revision,
                "delegate": decision["delegate"],
                "latency_ms": round(latency * 1000, 3),
                "request": request_record,
                "response": copy.deepcopy(data),
            }
            self.audit_log.append(classification_event)
            if not decision["delegate"]:
                state.last_revision = revision
                state.prior_entry_hashes = _entry_hashes(entries)
                attempts.append({
                    "agent": state.config.name,
                    "outcome": "declined",
                    "latency_ms": classification_event["latency_ms"],
                })
                continue

            work_payload = self._work_payload(state, recent_context)
            work_data = None
            work_latency = 0.0
            work_request = None
            try:
                work_data, work_latency, work_request = self._request(
                    state,
                    stage="work",
                    payload=work_payload,
                    timeout=WORK_TIMEOUT_SECONDS,
                    revision=revision,
                )
                response, raw_message = _normalize_tool_response(work_data)
                self._validate_calls(state, response)
                self._record_recovery(state, "work", revision)
            except Exception as error:
                if work_request is not None:
                    setattr(error, "_aux_request_record", work_request)
                    setattr(
                        error,
                        "_aux_response_record",
                        {"json": copy.deepcopy(work_data)},
                    )
                    setattr(error, "_aux_latency_s", work_latency)
                attempts.append(
                    self._record_failure(
                        state,
                        error,
                        stage="work",
                        revision=revision,
                    )
                )
                continue

            state.last_revision = revision
            state.prior_entry_hashes = _entry_hashes(entries)
            state.history.append({
                "type": "work_response",
                "context_revision": revision,
                "content": _message_content(raw_message),
                "tool_calls": copy.deepcopy(response["tool_calls"]),
            })
            self.audit_log.append({
                "event": "work",
                "agent": state.config.name,
                "base_url": state.config.base_url,
                "model": state.config.model,
                "context_revision": revision,
                "latency_ms": round(work_latency * 1000, 3),
                "request": work_request,
                "response": copy.deepcopy(work_data),
                "action": [call["name"] for call in response["tool_calls"]],
            })
            if not response["tool_calls"]:
                attempts.append({
                    "agent": state.config.name,
                    "outcome": "work_no_tool",
                    "latency_ms": round(work_latency * 1000, 3),
                })
                continue

            return AuxiliaryTurn(
                response=response,
                agent_name=state.config.name,
                context_revision=revision,
                escalation_packet=None,
            )

        packet = {
            "schema_version": 1,
            "type": "auxiliary_escalation",
            "context_revision": revision,
            "reason": "No auxiliary agent produced an authorized mechanical tool call.",
            "attempts": attempts,
            "recent_context": recent_context,
        }
        self.audit_log.append({
            "event": "escalation",
            "context_revision": revision,
            "escalation_reason": packet["reason"],
            "packet": copy.deepcopy(packet),
        })
        return AuxiliaryTurn(
            response=None,
            agent_name=None,
            context_revision=revision,
            escalation_packet=packet,
        )

    def record_tool_results(self, agent_name: str, execution_records: list[dict]):
        state = self._states_by_name.get(agent_name)
        if not state:
            return
        record = {
            "type": "tool_results",
            "results": copy.deepcopy(execution_records),
        }
        state.history.append(record)
        self.audit_log.append({
            "event": "tool_results",
            "agent": agent_name,
            "model": state.config.model,
            "action": [item.get("name") for item in execution_records],
            "tool_results": copy.deepcopy(execution_records),
        })

    @staticmethod
    def append_escalation_to_reminder(reminder: str, packet: dict) -> str:
        return (
            f"{reminder}\n\n"
            "AUXILIARY ESCALATION PACKET (harness metadata, not user-authored):\n"
            + json.dumps(packet, ensure_ascii=False, default=str)
        )
