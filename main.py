#!/usr/bin/env python3
"""AI E-Ink Friend — agent loop orchestrator."""

import time
import os
import signal
import sys
import json
import secrets
import copy
import hashlib
import socket
import ssl
import threading
import uuid
from contextlib import contextmanager
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import requests
from config import (
    PROJECT_DIR,
    DISPLAY_SERVER_URL,
    build_system_prompt,
    get_tool_definitions,
    MAX_TOOL_CALLS_PER_TURN,
    MIN_DISPLAY_INTERVAL,
    MIN_WAIT_SECONDS,
    MAX_WAIT_SECONDS,
    IDLE_TIMEOUT,
    BUTTON_CHECK_INTERVAL,
    CHAT_SERVER_PORT,
    CHAT_PASSWORD,
    CHAT_SESSION_DAYS,
    CHAT_USE_HTTPS,
    SSL_CERT_FILE,
    SSL_KEY_FILE,
    CHAT_MAX_IMAGES_PER_MESSAGE,
    CHAT_MAX_MEDIA_BYTES,
    CHAT_MAX_REQUEST_BYTES,
    CHAT_TAKEOVER_SECONDS,
    CHAT_SSE_MAX_STREAMS,
    CHAT_SSE_HEARTBEAT_SECONDS,
    CHAT_SSE_IDLE_SECONDS,
    REVIEW_INTERVAL,
    NOTIFICATION_APPROVAL_MODE,
    build_policy_reminder,
    estimate_tool_tokens,
    LLM_ESTIMATED_MAX_TOKENS,
    COMPACT_AFTER_N_MESSAGES,
    MERGE_SUMMARIES_AFTER,
    LLM_SUPPORTS_IMAGES,
    ENABLE_CAMERA,
    ENABLE_DISPLAY,
    ENABLE_ACTIVITY_LOG,
    VISION_PROVIDER,
    VISION_REQUESTS_GUIDANCE,
    VISION_POLL_INTERVAL,
    MOTION_POLL_INTERVAL,
    CHILL_TIMEOUT,
    SCENE_RMS_THRESHOLD,
    SCENE_PCT_THRESHOLD,
    SCENE_MAX_STALE_SECONDS,
    ENABLE_REOLINK,
    ENABLE_WEB_SEARCH,
    REOLINK_IP,
    REOLINK_USER,
    REOLINK_PASSWORD,
    REOLINK_TIMEOUT,
    POMODORO_WORK_MINUTES,
    POMODORO_IDLE_EXIT_SECONDS,
)
from notifications import NotificationStore
from caffeine import DrinkStore
from activity import ActivityStore, classify_scene
from pomodoro import PomodoroStore
from presence import ActiveTracker
from status_publisher import StatusPublisher
from context import Context
from ai_client import AIClient, LLMError, VisionClient
from vision_history import VisionDescriptionLog, VisionRequestHistory
from chat_media import (
    ChatMediaError,
    build_chat_message,
    media_data_from_message,
    merge_queued_messages,
    queued_message_payload,
    update_queued_text,
)
from reolink import ReoLinkCamera
from mcp_client import MCPClient
from sounds import play as play_sound
from tts import speak as tts_speak, interrupt as tts_interrupt
import logger
from logger import info


def http_get(path: str, timeout: int = 5) -> dict:
    try:
        r = requests.get(f"{DISPLAY_SERVER_URL}{path}", timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        info(f"[HTTP GET] {path}: {e}")
    return {}


def http_post(path: str, data: dict, timeout: int = 5):
    try:
        r = requests.post(f"{DISPLAY_SERVER_URL}{path}", json=data, timeout=timeout)
        return r.status_code == 200
    except Exception as e:
        info(f"[HTTP POST] {path}: {e}")
    return False


def _visible_response_text(response: dict) -> str:
    """Text the user will see from a model response, for media demotion context."""
    parts = []
    content = (response.get("content") or "").strip()
    if content:
        parts.append(content)
    for tool_call in response.get("tool_calls", []):
        if tool_call.get("name") not in ("update_display", "send_chat_message"):
            continue
        args = tool_call.get("arguments", {})
        text = args.get("text", "") if isinstance(args, dict) else ""
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    return "\n".join(dict.fromkeys(parts))


LONG_ACTION_TOOLS = {"take_photo", "capture_photo", "take_reolink_photo"}

TOOL_LABELS = {
    "take_photo": "checking the room camera",
    "capture_photo": "capturing a fresh room photo",
    "take_reolink_photo": "capturing a security-camera photo",
    "update_display": "updating the display",
    "send_chat_message": "sending a chat message",
    "update_vision_requests": "updating vision requests",
    "flash_ir_light": "adjusting infrared light",
    "flash_camera_light": "adjusting camera light",
    "log_drink": "logging a drink",
    "list_drinks": "checking the drink log",
    "edit_drink": "editing the drink log",
    "list_activity": "checking activity history",
    "log_pomodoro": "logging a pomodoro",
    "list_pomodoros": "checking pomodoro history",
    "edit_pomodoro": "editing pomodoro history",
    "pomodoro_stats": "calculating pomodoro stats",
    "enter_pomodoro_mode": "starting pomodoro mode",
    "exit_pomodoro_mode": "ending pomodoro mode",
    "propose_notification": "proposing a notification",
    "resolve_notification_proposal": "resolving a notification",
    "schedule_notification": "scheduling a notification",
    "delete_notification": "deleting a notification",
    "brave_web_search": "searching the web",
    "brave_local_search": "searching nearby places",
    "brave_image_search": "searching images",
    "brave_video_search": "searching videos",
    "brave_news_search": "searching the news",
    "brave_summarizer": "summarizing search results",
}

CHAT_STATIC_ASSETS = ("chat.css", "chat.mjs", "chat_model.mjs")


def _chat_asset_version(static_dir: str) -> str:
    """Return the restart-stable cache key for all transitive chat assets."""
    digest = hashlib.sha256()
    for filename in CHAT_STATIC_ASSETS:
        with open(os.path.join(static_dir, filename), "rb") as asset:
            digest.update(asset.read())
    return digest.hexdigest()[:12]


class ChatUIState:
    """Thread-safe agent state and chat invalidation publisher."""

    def __init__(self, activity_enabled: bool = False, activity: dict | None = None):
        self.condition = threading.Condition()
        self.server_id = uuid.uuid4().hex
        self.agent = {
            "mode": "sleeping",
            "detail": "",
            "locks_input": False,
            "revision": 0,
        }
        self.activity = {
            "enabled": bool(activity_enabled),
            "presence": None,
            "activity": None,
            "observed_at": None,
            "revision": 0,
        }
        if activity:
            self.activity.update({
                "presence": activity.get("presence"),
                "activity": activity.get("activity"),
                "observed_at": activity.get("last_observed_at"),
            })
        self.chat_revision = 0
        self.event_revision = 0

    def snapshot(self) -> dict:
        with self.condition:
            return self._snapshot_locked()

    def _snapshot_locked(self) -> dict:
        return {
            "agent": dict(self.agent),
            "activity": dict(self.activity),
            "server_id": self.server_id,
            "chat_revision": self.chat_revision,
            "event_revision": self.event_revision,
        }

    def set_agent(self, mode: str, detail: str = "", locks_input: bool = False):
        with self.condition:
            changed = (
                self.agent["mode"] != mode
                or self.agent["detail"] != detail
                or self.agent["locks_input"] != bool(locks_input)
            )
            if not changed:
                return
            self.agent = {
                "mode": mode,
                "detail": detail,
                "locks_input": bool(locks_input),
                "revision": self.agent["revision"] + 1,
            }
            self.event_revision += 1
            self.condition.notify_all()

    def set_activity(self, activity: dict | None):
        with self.condition:
            next_value = {
                "enabled": self.activity["enabled"],
                "presence": activity.get("presence") if activity else None,
                "activity": activity.get("activity") if activity else None,
                "observed_at": activity.get("last_observed_at") if activity else None,
            }
            changed = any(
                self.activity.get(key) != value for key, value in next_value.items()
            )
            if not changed:
                return
            next_value["revision"] = self.activity["revision"] + 1
            self.activity = next_value
            self.event_revision += 1
            self.condition.notify_all()

    def chat_changed(self):
        with self.condition:
            self.chat_revision += 1
            self.event_revision += 1
            self.condition.notify_all()

    def wait_after(self, revision: int, timeout: float) -> tuple[dict, bool]:
        with self.condition:
            if self.event_revision == revision:
                self.condition.wait(timeout)
            changed = self.event_revision != revision
            return self._snapshot_locked(), changed


class Orchestrator:
    def __init__(self):
        self.ctx = Context()
        if ENABLE_CAMERA:
            self.vision_request_history = VisionRequestHistory()
            self.vision_description_log = VisionDescriptionLog()
            info(
                "[VISION] Request history ready at "
                f"{self.vision_request_history.repo_dir} "
                f"(commit {self.vision_request_history.current_commit()})"
            )
        else:
            self.vision_request_history = None
            self.vision_description_log = None
        # Import camera dependencies lazily so camera-off chat mode stays light.
        # Camera itself selects picamera2 on Pi and OpenCV on macOS.
        if ENABLE_CAMERA:
            from camera import Camera
            self.camera = Camera()
        else:
            self.camera = None
        self.ai = AIClient()
        self.vision = (
            VisionClient(self.vision_request_history, self.vision_description_log)
            if ENABLE_CAMERA
            else None
        )
        self.reolink = ReoLinkCamera(REOLINK_IP, REOLINK_USER, REOLINK_PASSWORD, REOLINK_TIMEOUT) if ENABLE_REOLINK else None
        self.running = True
        self.last_display_time = 0
        self.chat_event = threading.Event()
        # Homogeneous dictionaries with stable queue IDs and optional media.
        self.chat_queue = []
        self.chat_queue_lock = threading.Lock()
        self.ctx_lock = threading.Lock()
        # Serializes cross-source snapshots and queue/context transfers without
        # making ordinary reads hold ctx_lock and chat_queue_lock together.
        self.chat_transfer_lock = threading.Lock()
        self.activity_store = ActivityStore() if ENABLE_ACTIVITY_LOG else None
        self.ui_state = ChatUIState(
            activity_enabled=ENABLE_ACTIVITY_LOG,
            activity=self.activity_store.current() if self.activity_store else None,
        )
        self.sse_slots = threading.BoundedSemaphore(max(1, CHAT_SSE_MAX_STREAMS))
        self.mcp_tools = []
        self.mcp = None
        self.notification_store = NotificationStore()
        # Used by smart notification approval to ensure the agent can only
        # resolve a proposal after a real chat response sent after its creation.
        self.last_chat_message_time = 0.0
        self.drink_store = DrinkStore()
        self.pomodoro_store = PomodoroStore()
        # Pomodoro mode: e-ink shows a focus screen (count + block end) and a
        # button press logs +1 cycle. See _render_pomodoro_screen / _tool_wait.
        self.pomodoro_mode = False
        self.pomodoro_work_minutes = POMODORO_WORK_MINUTES
        self.pomodoro_block_end = 0.0      # epoch seconds the current work block ends
        self.pomodoro_last_activity = 0.0  # last button/chat/motion while focusing
        self.pomodoro_idle_notified = False
        self.presence = ActiveTracker()
        self.status_publisher = StatusPublisher(self.drink_store, self.presence)
        self.last_review_time = time.time()
        self.active_notification = None  # {"id": str, "message": str, "remaining": int}

        # LLM error backoff state
        self._llm_failures = 0
        self._last_llm_fail = 0.0

        # Vision background thread state
        self.latest_scene = None  # description, timestamp, and vision request commit
        self.scene_lock = threading.Lock()
        # Acquire before capture so a request waiting behind another inference
        # takes a fresh frame instead of eventually submitting a stale one.
        self.vision_job_lock = threading.Lock()
        self.motion_event = threading.Event()
        self.motion_description = ""
        if ENABLE_CAMERA:
            from scene_change import SceneChangeDetector
            self.scene_detector = SceneChangeDetector(
                rms_threshold=SCENE_RMS_THRESHOLD,
                pct_threshold=SCENE_PCT_THRESHOLD,
                max_stale_seconds=SCENE_MAX_STALE_SECONDS,
            )
        else:
            self.scene_detector = None
        self.vision_mode = "chill"  # "chill" when no motion, "active" when motion detected
        self.vision_requests_shown = False  # tracks if we've shown existing requests this turn

        # Chat auth — persist session token across restarts
        self._token_file = os.path.join(PROJECT_DIR, ".session_token")
        self.session_token = ""
        if os.path.exists(self._token_file):
            try:
                with open(self._token_file, "r") as f:
                    self.session_token = f.read().strip()
                if self.session_token:
                    info(f"[AUTH] Loaded existing session token")
            except Exception:
                self.session_token = ""
        if not self.session_token:
            self.session_token = secrets.token_hex(32)
            with open(self._token_file, "w") as f:
                f.write(self.session_token)
            info(f"[AUTH] Generated new session token")
        info(f"[AUTH] Password: {'***' if CHAT_PASSWORD != 'admin' else 'admin (default)'}, session lasts {CHAT_SESSION_DAYS} days")

        if ENABLE_WEB_SEARCH:
            try:
                info("Init MCP client...")
                self.mcp = MCPClient()
                tools = self.mcp.initialize()
                self.mcp_tools = self.mcp.get_tool_definitions()
                info(f"[MCP] Discovered {len(tools)} tools: {[t['name'] for t in tools]}")
            except Exception as e:
                info(f"[MCP] Unavailable: {e}")
        else:
            info("[MCP] Web search disabled (ENABLE_WEB_SEARCH=0)")

        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        info("\nShutting down...")
        self.running = False
        with self.ui_state.condition:
            self.ui_state.condition.notify_all()

    def _set_agent_state(self, mode: str, detail: str = "", locks_input: bool = False):
        self.ui_state.set_agent(mode, detail, locks_input)

    @contextmanager
    def _blocked_agent_state(self, detail: str, locks_input: bool):
        previous = self.ui_state.snapshot()["agent"]
        self._set_agent_state("blocked", detail, locks_input)
        try:
            yield
        finally:
            self._set_agent_state(
                previous["mode"],
                previous["detail"],
                previous["locks_input"],
            )

    def _llm_backoff_sleep(self, seconds: float):
        with self._blocked_agent_state("LLM backoff after failures", False):
            time.sleep(seconds)

    @staticmethod
    def _chat_time(ts: float | None) -> str:
        if not ts:
            return ""
        return time.strftime("%-I:%M%p %a", time.localtime(ts)).lower().lstrip("0")

    def _queue_bubble(self, entry: dict) -> dict:
        images = [
            {
                "url": f"/chat/media/{attachment['id']}",
                "name": attachment.get("name", "image"),
                "type": attachment.get("type", ""),
            }
            for attachment in entry.get("chat_images", [])
            if isinstance(attachment, dict) and attachment.get("id")
        ]
        return {
            "id": entry["id"],
            "role": "user",
            "content": str(entry.get("text", "")),
            "images": images,
            "time": self._chat_time(entry.get("created_at")),
            "queued": True,
        }

    def _sync_chat_event_locked(self):
        if self.chat_queue:
            self.chat_event.set()
        else:
            self.chat_event.clear()

    def _snapshot_chat_sources(self) -> tuple[list, list]:
        """Return a transfer-consistent context/queue snapshot.

        The transaction gate prevents a sweep between the sequential copies, so
        this read path never holds both data locks at once.
        """
        with self.chat_transfer_lock:
            with self.ctx_lock:
                context_messages = copy.deepcopy(self.ctx.messages)
            with self.chat_queue_lock:
                queued_messages = copy.deepcopy(self.chat_queue)
        return context_messages, queued_messages

    def _find_chat_media(self, media_id: str):
        with self.chat_transfer_lock:
            with self.ctx_lock:
                for msg in reversed(self.ctx.messages):
                    found = media_data_from_message(msg, media_id)
                    if found:
                        return found
            with self.chat_queue_lock:
                for entry in reversed(self.chat_queue):
                    queued_msg = {
                        "content": entry.get("content", []),
                        "_chat_images": entry.get("chat_images", []),
                    }
                    found = media_data_from_message(queued_msg, media_id)
                    if found:
                        return found
        return None

    def _undo_queued_message(self, queue_id: str) -> dict | None:
        restored = None
        with self.chat_transfer_lock:
            with self.chat_queue_lock:
                for index, entry in enumerate(self.chat_queue):
                    if entry.get("id") == queue_id:
                        removed = self.chat_queue.pop(index)
                        # Serialize while the transaction gate still excludes a
                        # competing sweep or media lookup. The response remains
                        # self-contained after the queue record disappears.
                        restored = queued_message_payload(removed)
                        self._sync_chat_event_locked()
                        break
        if restored is not None:
            self.ui_state.chat_changed()
            return restored
        return None

    def _edit_queued_message(self, queue_id: str, message: str) -> dict | None:
        updated = None
        with self.chat_transfer_lock:
            with self.chat_queue_lock:
                for entry in self.chat_queue:
                    if entry.get("id") == queue_id:
                        update_queued_text(entry, message)
                        updated = copy.deepcopy(entry)
                        break
        if updated is not None:
            self.ui_state.chat_changed()
            return self._queue_bubble(updated)
        return None

    def _sweep_chat_queue(self) -> str | None:
        """Merge the queue into one context message at a wait boundary."""
        with self.chat_transfer_lock:
            with self.chat_queue_lock:
                boundary_entries = copy.deepcopy(self.chat_queue)
                boundary_ids = [entry.get("id") for entry in boundary_entries]
                if not boundary_entries:
                    self._sync_chat_event_locked()
                    return None

            content, chat_images, visible_text, dropped = merge_queued_messages(
                boundary_entries
            )

            with self.ctx_lock:
                with self.chat_queue_lock:
                    current_ids = [
                        entry.get("id")
                        for entry in self.chat_queue[:len(boundary_ids)]
                    ]
                    if current_ids != boundary_ids:
                        raise RuntimeError("chat queue changed during sweep preparation")
                    del self.chat_queue[:len(boundary_ids)]
                    context_id = self.ctx.add_user(
                        content,
                        chat_images=chat_images,
                        chat_original_text=visible_text,
                    )
                    self._sync_chat_event_locked()

        info(
            f"[CHAT] Swept {len(boundary_entries)} queued message(s) into "
            f"{context_id}; dropped {dropped} image(s)"
        )
        self.ui_state.chat_changed()
        return context_id

    def run(self):
        if self.camera:
            info(f"Init camera ({self.camera.backend})...")
        else:
            info("Camera disabled.")
        info(f"Init AI client ({self.ai.model} on OpenRouter)...")
        if self.vision:
            info(f"Init vision client ({self.vision.model} via {self.vision.provider})...")
            if self.vision.provider == "aarg_mlx":
                state = "ready" if self.vision.health_check() else "not ready"
                info(f"[VISION] AARG service is {state} at {self.vision.base_url}")
        self._start_chat_server()
        self.status_publisher.start()
        if ENABLE_CAMERA:
            self._start_vision_loop()
        with self.ctx_lock:
            if self.ctx.load():
                info("Resuming from saved context.")
                # Always refresh system prompt to pick up changes
                prompt = build_system_prompt()
                if self.ctx.messages and self.ctx.messages[0].get("role") == "system":
                    self.ctx.messages[0]["content"] = prompt
                    info("[CONTEXT] Refreshed system prompt in loaded context.")
                else:
                    self.ctx.messages.insert(0, self.ctx.new_message("system", prompt))
                    info("[CONTEXT] Inserted system prompt into loaded context.")
                if ENABLE_CAMERA:
                    self.ctx.add_user("A restart just occurred. Resume where you left off.")
                else:
                    self.ctx.add_user("A restart just occurred. Camera is not available — resume where you left off.")
            else:
                self.ctx.add_system(build_system_prompt())
                if ENABLE_CAMERA:
                    self.ctx.add_user("You just woke up! Use take_photo to see the room and say hi.")
                else:
                    self.ctx.add_user("You just woke up! Note: camera/vision tools are not available. Use your other tools to say hi.")
        self.ui_state.chat_changed()
        info("Entering agent loop.")

        while self.running:
            try:
                self._turn()
            except LLMError as e:
                info(f"[FATAL] Unhandled LLM error: {e}")
                self._llm_failures += 1
                delay = self._llm_backoff_seconds()
                info(f"[FATAL] Backing off for {delay}s")
                self._llm_backoff_sleep(delay)
            except Exception as e:
                info(f"[ERROR] {e}")
                time.sleep(5)

        self.cleanup()

    def _turn(self):
        while self.running:
            tools = list(get_tool_definitions())
            if self.mcp_tools:
                tools.extend(self.mcp_tools)

            with self.ctx_lock:
                self.ctx.demote_old_images()
                self.ctx._repair_pairing()
                messages = self.ctx.get_messages()
                msg_tokens = self.ctx.total_tokens()
            reminder = self._build_turn_reminder()
            messages.append({"role": "user", "content": reminder})
            estimated = msg_tokens + estimate_tool_tokens(tools) + len(reminder) // 4
            if estimated > LLM_ESTIMATED_MAX_TOKENS:
                info(f"[LLM] Token estimate {estimated} exceeds limit {LLM_ESTIMATED_MAX_TOKENS}, compacting...")
                with self._blocked_agent_state("compacting memory", True):
                    try:
                        self.ctx.check_compact(self.ai, self.ctx_lock)
                    except LLMError as e:
                        info(f"[LLM] Compaction failed during token overflow: {e}")
                self.ui_state.chat_changed()
                with self.ctx_lock:
                    messages = self.ctx.get_messages()
                    msg_tokens = self.ctx.total_tokens()
                reminder = self._build_turn_reminder()
                messages.append({"role": "user", "content": reminder})
                estimated = msg_tokens + estimate_tool_tokens(tools) + len(reminder) // 4
                info(f"[LLM] After compaction: ~{msg_tokens} msg tokens + {estimate_tool_tokens(tools)} tool tokens = ~{estimated} total")
            info(f"[LLM] Sending {len(messages)} messages (~{msg_tokens} msg tokens, ~{estimate_tool_tokens(tools)} tool tokens, ~{estimated} total)...")
            play_sound("thinking")
            self._set_agent_state("thinking")
            try:
                response = self.ai.chat_with_tools(messages, tools)
            except LLMError as e:
                recoverable = e.status_code >= 500 or e.status_code == 429
                err_str = str(e)
                if "exceed_context_size_error" in err_str or "exceeds the available context size" in err_str:
                    info(f"[LLM] Context overflow detected, triggering compaction...")
                    with self._blocked_agent_state("compacting memory", True):
                        try:
                            self.ctx.check_compact(self.ai, self.ctx_lock)
                        except LLMError as ce:
                            info(f"[LLM] Compaction failed during overflow: {ce}")
                    self.ui_state.chat_changed()
                    time.sleep(1)
                    continue
                if not recoverable:
                    info(f"[LLM] Non-recoverable error (HTTP {e.status_code}), backing off: {e}")
                    self._llm_failures += 1
                    self._last_llm_fail = time.time()
                    delay = self._llm_backoff_seconds()
                    info(f"[LLM] Backing off for {delay}s ({self._llm_failures} consecutive failures)")
                    self._display_error(f"LLM API error ({e.status_code}). Retrying in {delay // 60}m...")
                    self._llm_backoff_sleep(delay)
                    continue
                else:
                    info(f"[LLM] Recoverable error (HTTP {e.status_code}), backing off: {e}")
                    self._llm_failures += 1
                    self._last_llm_fail = time.time()
                    delay = min(self._llm_backoff_seconds(), 120)  # cap transient retries at 2min
                    info(f"[LLM] Backing off for {delay}s ({self._llm_failures} consecutive failures)")
                    self._llm_backoff_sleep(delay)
                    continue
            except (requests.Timeout, requests.ConnectionError) as e:
                info(f"[LLM] Network error: {e}")
                self._llm_failures += 1
                self._last_llm_fail = time.time()
                delay = min(self._llm_backoff_seconds(), 120)
                info(f"[LLM] Backing off for {delay}s")
                self._llm_backoff_sleep(delay)
                continue
            except Exception as e:
                info(f"[LLM] Unexpected error: {e}")
                self._llm_failures += 1
                self._last_llm_fail = time.time()
                self._llm_backoff_sleep(5)
                continue

            self._llm_failures = 0

            with self.ctx_lock:
                self.ctx.add_assistant(response)
                self.ctx.note_latest_image_response(
                    _visible_response_text(response)
                )
            if response["tool_calls"]:
                self.ui_state.chat_changed()

            if response["reasoning"]:
                info(f"[REASONING] {response['reasoning'][:200]}...")
                info(f"[REASONING] {response['reasoning']}")
            if response["content"]:
                info(f"[AI] {response['content'][:200]}")
                info(f"[AI] {response['content']}")

            if not response["tool_calls"]:
                # If the brain model returned text but no tool call, display it automatically
                if response["content"]:
                    content = response["content"]
                    if len(content) > 140:
                        info(f"[AUTO-CHAT] AI returned long content without tool call, sending to chat...")
                        result = self._execute_tool("send_chat_message", {"text": content})
                    else:
                        info(f"[AUTO-DISPLAY] AI returned content without update_display, showing it...")
                        result = self._execute_tool("update_display", {"text": content})
                    if result.get("status") == "ok":
                        self._execute_tool("wait", {})
                        self._sweep_chat_queue()
                    continue
                info("[IDLE] AI produced no tool calls. Waiting...")
                self._idle_wait()
                return

            self._execute_tool_batch(response["tool_calls"])

            try:
                with self.ctx_lock:
                    self.ctx.demote_old_images()
                    will_compact = len(self.ctx.messages) >= COMPACT_AFTER_N_MESSAGES
                if will_compact:
                    with self._blocked_agent_state("compacting memory", True):
                        self.ctx.check_compact(self.ai, self.ctx_lock)
                    self.ui_state.chat_changed()
                else:
                    self.ctx.check_compact(self.ai, self.ctx_lock)
                # Only merge summaries when user is away (chill mode) to avoid
                # blocking the agent loop with back-to-back LLM calls
                if self.vision_mode == "chill" or not ENABLE_CAMERA:
                    with self.ctx_lock:
                        will_merge = sum(
                            1 for msg in self.ctx.messages if self.ctx._is_summary(msg)
                        ) > MERGE_SUMMARIES_AFTER
                    if will_merge:
                        with self._blocked_agent_state("merging memory", True):
                            self.ctx.check_merge_summaries(self.ai, self.ctx_lock)
                        self.ui_state.chat_changed()
            except LLMError as e:
                info(f"[LLM] Compaction failed: {e}")

    def _execute_tool_batch(self, tool_calls: list[dict]):
        """Execute a complete model-emitted batch, then honor a wait sweep."""
        deferred_user_msgs = []
        saw_wait = False

        for tool_call in tool_calls:
            name = tool_call["name"]
            if name == "wait":
                saw_wait = True
            info(f"[TOOL] {name}({tool_call['arguments']})")
            try:
                result = self._execute_tool(name, tool_call["arguments"])
            except Exception as error:
                result = {
                    "status": "error",
                    "message": f"Tool execution failed: {error}",
                }
                info(f"[TOOL ERROR] {error}")
            info(f"[TOOL RESULT] {json.dumps(result)[:200]}")
            info(f"[TOOL RESULT] {json.dumps(result)}")
            with self.ctx_lock:
                self.ctx.add_tool_result(tool_call["id"], name, result)
            user_message = result.get("user_message")
            if user_message:
                deferred_user_msgs.append(user_message)

        if deferred_user_msgs:
            with self.ctx_lock:
                for message in deferred_user_msgs:
                    self.ctx.add_user(message)

        # No call is skipped or synthesized. The sweep happens only after every
        # real result has closed the assistant/tool pairing.
        if saw_wait:
            self._sweep_chat_queue()

    def _llm_backoff_seconds(self) -> int:
        n = max(self._llm_failures - 1, 0)
        return min(30 * (2 ** n), 1800)

    def _display_error(self, msg: str):
        if not ENABLE_DISPLAY:
            return  # chat-only: no e-ink to show an error banner on
        try:
            requests.post(
                f"{DISPLAY_SERVER_URL}/display",
                json={"text": msg},
                timeout=5,
            )
        except Exception:
            pass

    def _execute_tool(self, name: str, args: dict) -> dict:
        if name == "wait":
            self._set_agent_state("sleeping")
        else:
            mode = "acting_long" if name in LONG_ACTION_TOOLS else "acting"
            self._set_agent_state(mode, TOOL_LABELS.get(name, name))
        try:
            return self._dispatch_tool(name, args)
        finally:
            self._set_agent_state("thinking")

    def _dispatch_tool(self, name: str, args: dict) -> dict:
        if name == "take_photo":
            if not ENABLE_CAMERA:
                return {"error": "Camera is disabled. Use other tools instead."}
            play_sound("take_photo")
            return self._tool_take_photo()
        elif name == "capture_photo":
            if not ENABLE_CAMERA:
                return {"error": "Camera is disabled. Use other tools instead."}
            play_sound("take_photo")
            return self._tool_capture_photo()
        elif name == "update_display":
            play_sound("update_display")
            return self._tool_update_display(args)
        elif name == "send_chat_message":
            play_sound("update_display")
            return self._tool_send_chat_message(args)
        elif name == "wait":
            play_sound("wait")
            return self._tool_wait(args)
        elif name == "update_vision_requests":
            return self._tool_update_vision_requests(args)
        elif name == "take_reolink_photo":
            if not ENABLE_REOLINK:
                return {"error": "Reolink camera is disabled."}
            play_sound("take_photo")
            return self._tool_take_reolink_photo()
        elif name == "flash_ir_light":
            if not ENABLE_REOLINK:
                return {"error": "Reolink camera is disabled."}
            return self._tool_flash_ir_light(args)
        elif name == "flash_camera_light":
            if not ENABLE_REOLINK:
                return {"error": "Reolink camera is disabled."}
            return self._tool_flash_camera_light(args)
        elif name == "log_drink":
            return self._tool_log_drink(args)
        elif name == "list_drinks":
            return self._tool_list_drinks(args)
        elif name == "edit_drink":
            return self._tool_edit_drink(args)
        elif name == "list_activity":
            return self._tool_list_activity(args)
        elif name == "log_pomodoro":
            return self._tool_log_pomodoro(args)
        elif name == "list_pomodoros":
            return self._tool_list_pomodoros(args)
        elif name == "edit_pomodoro":
            return self._tool_edit_pomodoro(args)
        elif name == "pomodoro_stats":
            return self._tool_pomodoro_stats(args)
        elif name == "enter_pomodoro_mode":
            return self._tool_enter_pomodoro_mode(args)
        elif name == "exit_pomodoro_mode":
            return self._tool_exit_pomodoro_mode(args)
        elif name == "propose_notification":
            play_sound("update_display")
            return self._tool_propose_notification(args)
        elif name == "resolve_notification_proposal":
            return self._tool_resolve_notification_proposal(args)
        elif name == "schedule_notification":
            return self._tool_schedule_notification(args)
        elif name == "delete_notification":
            return self._tool_delete_notification(args)
        else:
            if self.mcp:
                play_sound("search")
                try:
                    return self.mcp.call_tool(name, args)
                except Exception as e:
                    return {"error": f"MCP tool '{name}' failed: {e}"}
            return {"error": f"Unknown tool: {name}. Available: take_photo, capture_photo, update_display, wait"}

    def _tool_take_photo(self) -> dict:
        # Wait up to 90s for the background vision thread to produce a scene
        for _ in range(90):
            with self.scene_lock:
                scene = self.latest_scene
            if scene and scene.get("description"):
                break
            time.sleep(1)
        else:
            return {"status": "error", "message": "No scene available yet — vision thread may still be starting"}

        captured_at = time.strftime("%-I:%M%p", time.localtime(scene["timestamp"])).lower().lstrip("0")
        age = int(time.time() - scene["timestamp"])
        result = {
            "status": "ok",
            "description": scene["description"],
            "captured_at": captured_at,
            "age_seconds": age,
            "vision_request_commit": scene.get("request_commit", ""),
        }
        self._note_stale_vision_requests(result)
        return result

    def _note_stale_vision_requests(self, result: dict):
        """Tell the agent when its stored requests were skipped as stale."""
        stale = getattr(self.vision, "requests_stale", "")
        if not stale:
            return
        result["vision_requests"] = (
            f"Not applied. They were written for {stale} vision, but this "
            f"machine now runs {VISION_PROVIDER}. Call update_vision_requests "
            "to rewrite them for the current mode."
        )

    def _tool_capture_photo(self) -> dict:
        """Take a photo now and block until the vision model describes it."""
        info("[PHOTO] Synchronous capture + describe (blocking, may take up to 120s)...")
        scene = self._capture_and_describe(source="main_camera_on_demand")
        if not scene:
            return {"status": "error", "message": "Failed to capture or describe photo — vision model may be unavailable"}
        captured_at = time.strftime("%-I:%M%p", time.localtime(scene["timestamp"])).lower().lstrip("0")
        result = {
            "status": "ok",
            "description": scene["description"],
            "captured_at": captured_at,
            "vision_request_commit": scene.get("request_commit", ""),
        }
        self._note_stale_vision_requests(result)
        return result

    def _tool_update_vision_requests(self, args: dict) -> dict:
        requests_text = args.get("requests", "").strip()
        if not requests_text:
            return {"status": "error", "message": "No requests text provided"}

        if not self.vision_request_history:
            return {"status": "error", "message": "Vision request history is unavailable"}

        # Read current contents so the AI can see what's already there.
        declared, current = self.vision_request_history.split_mode(
            self.vision_request_history.read()
        )
        stale = bool(current) and declared != VISION_PROVIDER

        # First call with existing content: bounce back so the AI can merge.
        # Requests written for another vision mode are the exception: merging
        # into those would carry the wrong shape forward, so let the rewrite go
        # straight through.
        if current and not stale and not self.vision_requests_shown:
            self.vision_requests_shown = True
            info(f"[VISION] Bouncing update_vision_requests — showing existing requests first")
            return {
                "status": "needs_retry",
                "message": (
                    "STOP — the vision requests file already has content. "
                    "Review the existing requests below and call update_vision_requests again "
                    "with your new requests MERGED with the existing ones. "
                    "Don't drop existing requests unless they're truly no longer needed. "
                    + VISION_REQUESTS_GUIDANCE
                ),
                "current_requests": current,
                "vision_mode": VISION_PROVIDER,
            }

        try:
            commit_hash = self.vision_request_history.update(requests_text, VISION_PROVIDER)
            self.vision_requests_shown = False  # reset so next update bounces again
            info(f"[VISION] Requests updated at {commit_hash}: {requests_text[:100]}...")
            return {
                "status": "ok",
                "message": (
                    f"Vision requests updated and committed for {VISION_PROVIDER} "
                    "vision. Changes take effect on the next photo capture."
                ),
                "commit_hash": commit_hash,
                "vision_mode": VISION_PROVIDER,
            }
        except Exception as e:
            return {"status": "error", "message": f"Failed to write requests file: {e}"}

    def _tool_take_reolink_photo(self) -> dict:
        if not self.reolink:
            return {"status": "error", "message": "Reolink camera not initialized"}
        if not self.vision:
            return {"status": "error", "message": "Vision model not available (ENABLE_CAMERA=0)"}
        info("[REOLINK] Capturing snapshot...")
        with self.vision_job_lock:
            try:
                _, data_uri = self.reolink.capture()
                captured_at_epoch = time.time()
            except Exception as e:
                return {"status": "error", "message": f"Reolink capture failed: {e}"}
            try:
                description = self.vision.describe(
                    data_uri,
                    source="reolink_security_cam",
                    captured_at=captured_at_epoch,
                )
            except Exception as e:
                return {"status": "error", "message": f"Vision describe failed: {e}"}
        if not description:
            return {"status": "error", "message": "Vision model returned empty description"}
        captured_at = time.strftime("%-I:%M%p").lower().lstrip("0")
        info(f"[REOLINK] Scene: {description[:100]}...")
        return {
            "status": "ok",
            "description": description,
            "captured_at": captured_at,
            "source": "reolink_security_cam",
            "vision_request_commit": self.vision.last_request_commit,
        }

    def _tool_flash_ir_light(self, args: dict) -> dict:
        if not self.reolink:
            return {"status": "error", "message": "Reolink camera not initialized"}
        state = args.get("state", "Auto")
        duration = args.get("duration_seconds")
        info(f"[REOLINK] IR light: state={state}, duration={duration}")
        try:
            success = self.reolink.set_ir_light(state)
        except Exception as e:
            return {"status": "error", "message": f"Failed to control IR light: {e}"}
        if not success:
            return {"status": "error", "message": "Camera returned an error"}
        if state == "Off" and duration:
            def revert():
                time.sleep(int(duration))
                try:
                    self.reolink.set_ir_light("Auto")
                    info("[REOLINK] IR light reverted to Auto")
                except Exception:
                    pass
            threading.Thread(target=revert, daemon=True).start()
            return {"status": "ok", "message": f"IR light set to {state}, reverting to Auto in {duration}s"}
        return {"status": "ok", "message": f"IR light set to {state}"}

    def _tool_flash_camera_light(self, args: dict) -> dict:
        if not self.reolink:
            return {"status": "error", "message": "Reolink camera not initialized"}
        on = bool(args.get("on", True))
        brightness = int(args.get("brightness", 100))
        duration = args.get("duration_seconds")
        info(f"[REOLINK] Flash light: on={on}, brightness={brightness}, duration={duration}")
        try:
            success = self.reolink.set_white_light(on, brightness)
        except Exception as e:
            return {"status": "error", "message": f"Failed to control light: {e}"}
        if not success:
            return {"status": "error", "message": "Camera returned an error — check credentials or model support"}
        if on and duration:
            def turn_off():
                time.sleep(int(duration))
                try:
                    self.reolink.set_white_light(False)
                    info("[REOLINK] Flash auto-off after duration")
                except Exception:
                    pass
            threading.Thread(target=turn_off, daemon=True).start()
            return {"status": "ok", "message": f"Light on at {brightness}% — will turn off in {duration}s"}
        return {"status": "ok", "message": f"Light {'on' if on else 'off'} at {brightness}% brightness"}

    def _capture_and_describe(self, source: str = "main_camera") -> dict | None:
        """Capture a photo and get a text description from the local vision model."""
        with self.vision_job_lock:
            try:
                jpeg_bytes, photo_uri = self.camera.capture()
                captured_at = time.time()
            except Exception as e:
                info(f"[VISION] Camera error: {e}")
                return None

            self._save_debug_image(jpeg_bytes)

            try:
                description = self.vision.describe(
                    photo_uri,
                    source=source,
                    captured_at=captured_at,
                )
            except Exception as e:
                info(f"[VISION] Describe error: {e}")
                return None
        if not description:
            info("[VISION] Got empty description from vision model, skipping")
            return None
        scene = {
            "description": description,
            "timestamp": captured_at,
            "request_commit": self.vision.last_request_commit,
        }
        with self.scene_lock:
            self.latest_scene = scene
        self._record_activity(description, captured_at, source)
        return scene

    def _record_activity(self, description: str, captured_at: float, source: str):
        """Classify and persist a main-camera description without involving the brain."""
        if not self.activity_store:
            return
        try:
            observation = classify_scene(description)
            result = self.activity_store.observe(
                observation,
                observed_at=captured_at,
                source=source,
            )
            current = result.get("current")
            if current:
                self.ui_state.set_activity(current)
            if result.get("changed") and current:
                info(
                    f"[ACTIVITY] {current['presence']} / "
                    f"{current['activity']} from {source}"
                )
        except Exception as exc:
            info(f"[ACTIVITY] Observation error: {exc}")

    def _save_debug_image(self, jpeg_bytes: bytes):
        """Save captured image to debug_images/, prune files older than 24h."""
        import os as _os
        import glob as _glob
        debug_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "debug_images")
        _os.makedirs(debug_dir, exist_ok=True)
        filename = time.strftime("%Y%m%d_%H%M%S") + ".jpg"
        try:
            with open(_os.path.join(debug_dir, filename), "wb") as f:
                f.write(jpeg_bytes)
        except Exception as e:
            info(f"[VISION] Failed to save debug image: {e}")
            return
        cutoff = time.time() - 86400
        for old in _glob.glob(_os.path.join(debug_dir, "*.jpg")):
            try:
                if _os.path.getmtime(old) < cutoff:
                    _os.remove(old)
            except Exception:
                pass

    def _interruptible_sleep(self, seconds):
        """Sleep for up to `seconds`, checking self.running each second."""
        for _ in range(int(seconds)):
            if not self.running:
                return
            time.sleep(1)

    def _start_vision_loop(self):
        from PIL import Image as _Image

        def loop():
            mode = "chill"
            last_vision_time = 0.0
            last_motion_time = 0.0
            info(f"[VISION] Motion loop started (poll={MOTION_POLL_INTERVAL}s, "
                 f"vision_cooldown={VISION_POLL_INTERVAL}s, chill_timeout={CHILL_TIMEOUT}s)")

            while self.running:
                # Fast tier: cheap lores capture for motion detection
                try:
                    gray = self.camera.capture_lores()
                except Exception as e:
                    info(f"[VISION] Lores capture error: {e}")
                    self._interruptible_sleep(10)
                    continue

                pil = _Image.fromarray(gray.astype(int).clip(0, 255).astype('uint8'), mode="L")
                result = self.scene_detector.check(pil)

                now = time.time()

                if result["changed"]:
                    last_motion_time = now
                    self.presence.touch()
                    info(f"[VISION] Motion detected ({mode}): {result['reason']} "
                         f"(rms={result['rms']:.1f}, pct={result['pct_changed']:.3f}, "
                         f"shift={result['shift']})")

                    if mode == "chill":
                        # First motion after quiet period: immediate capture + notify agent
                        mode = "active"
                        self.vision_mode = "active"
                        info("[VISION] Entering active mode (was chill)")
                        self._do_vision_capture(notify_agent=True)
                        last_vision_time = time.time()
                    elif now - last_vision_time >= VISION_POLL_INTERVAL:
                        # Active mode but cooldown elapsed: refresh vision
                        self._do_vision_capture(notify_agent=False)
                        last_vision_time = time.time()
                else:
                    # No motion
                    if mode == "active" and now - last_motion_time > CHILL_TIMEOUT:
                        mode = "chill"
                        self.vision_mode = "chill"
                        info("[VISION] Entering chill mode (no motion for "
                             f"{CHILL_TIMEOUT}s)")

                self._interruptible_sleep(MOTION_POLL_INTERVAL)

            info("[VISION] Motion loop stopped")

        t = threading.Thread(target=loop, daemon=True)
        t.start()

    def _do_vision_capture(self, notify_agent=False):
        """Full-res capture + vision model describe. Optionally interrupt the agent."""
        scene = self._capture_and_describe(source="main_camera_background")
        if not scene:
            return
        description = scene["description"]
        info(f"[VISION] Scene updated: {description[:100]}...")

        if notify_agent:
            self.motion_description = description
            self.motion_event.set()

    def _tool_update_display(self, args: dict, speak: bool = True) -> dict:
        text = args.get("text", "")

        if not text.strip():
            return {"status": "error", "message": "No text provided"}

        if speak:
            tts_speak(text)

        # Chat-only mode: there is no e-ink display. The message already
        # surfaces in the web chat (the chat UI renders update_display tool
        # calls), so just voice it and return — no display server round-trip.
        if not ENABLE_DISPLAY:
            return {"status": "ok", "message": "Message sent to chat."}

        # Pomodoro mode owns the e-ink (the focus screen). Don't overwrite it —
        # the text still reaches the user via voice and the web chat log.
        if self.pomodoro_mode:
            return {"status": "ok", "message": "Sent to chat/voice. The e-ink keeps showing the pomodoro focus screen."}

        timestamp = time.strftime("%-I:%M%p").lower().lstrip("0")
        text = f"{text}\n\n— {timestamp}"

        elapsed = time.monotonic() - self.last_display_time
        if elapsed < MIN_DISPLAY_INTERVAL:
            time.sleep(MIN_DISPLAY_INTERVAL - elapsed)

        success = http_post("/display", {"text": text}, timeout=15)
        self.last_display_time = time.monotonic()

        if success:
            return {"status": "ok", "message": "Display updated."}
        else:
            return {"status": "error", "message": "Failed to communicate with display server"}

    def _tool_send_chat_message(self, args: dict) -> dict:
        text = args.get("text", "")
        if not text.strip():
            return {"status": "error", "message": "No text provided"}

        tts_speak(text)

        # Chat-only mode: the full message already renders in the chat UI, so
        # there is no e-ink preview to show.
        if not ENABLE_DISPLAY:
            return {"status": "ok", "message": "Chat message sent."}

        # Pomodoro mode owns the e-ink (the focus screen) — skip the preview.
        if self.pomodoro_mode:
            return {"status": "ok", "message": "Chat message sent. The e-ink keeps showing the pomodoro focus screen."}

        # Show a preview on the e-ink display
        preview_max = 90
        if len(text) <= preview_max:
            preview = text
        else:
            preview = text[:preview_max].rsplit(" ", 1)[0] + "..."
        display_text = f"{preview}\n(full message on chat)"
        self._tool_update_display({"text": display_text}, speak=False)

        return {"status": "ok", "message": "Chat message sent and display preview shown."}

    def _tool_wait(self, args: dict) -> dict:
        seconds = max(MIN_WAIT_SECONDS, min(MAX_WAIT_SECONDS, int(args.get("seconds", 60))))
        info(f"[WAIT] Sleeping {seconds}s...")

        start = time.monotonic()
        while time.monotonic() - start < seconds:
            if self.chat_event.is_set():
                self._note_pomodoro_activity()
                waited = int(time.monotonic() - start)
                info(f"[WAIT] Interrupted by chat message after {waited}s")
                return {"status": "interrupted", "reason": "chat_message", "waited": waited}

            if self.motion_event.is_set():
                self.motion_event.clear()
                self._note_pomodoro_activity()
                waited = int(time.monotonic() - start)
                desc = self.motion_description or "Something moved"
                info(f"[WAIT] Interrupted by motion after {waited}s")
                return {"status": "interrupted", "reason": "motion_detected", "waited": waited,
                        "user_message": f"Motion detected in the room! Here's what the camera sees: {desc}"}

            # In pomodoro mode, if he's gone quiet with no button/chat/motion for
            # the idle window, nudge the agent once to decide (from context) whether
            # to exit. The agent, not the loop, makes the call.
            if (self.pomodoro_mode and not self.pomodoro_idle_notified
                    and time.time() - self.pomodoro_last_activity > POMODORO_IDLE_EXIT_SECONDS
                    and not self.presence.is_active()):
                self.pomodoro_idle_notified = True
                waited = int(time.monotonic() - start)
                mins = POMODORO_IDLE_EXIT_SECONDS // 60
                info(f"[POMODORO] Idle {mins}min in pomodoro mode — nudging agent to decide")
                return {
                    "status": "interrupted", "reason": "pomodoro_idle", "waited": waited,
                    "user_message": (
                        f"You've been in pomodoro mode for {mins} min with no button press, "
                        "chat, or motion. If Austin seems to have left or is done, call "
                        "exit_pomodoro_mode. If he might just be heads-down working, leave it on."
                    ),
                }

            if not self.running:
                return {"status": "interrupted", "reason": "shutdown", "waited": int(time.monotonic() - start)}

            if time.time() - self.last_review_time > REVIEW_INTERVAL:
                patterns = self._detect_patterns()
                summary = self.notification_store.get_review_summary(
                    patterns=patterns
                )
                self.last_review_time = time.time()
                waited = int(time.monotonic() - start)
                info(f"[WAIT] Interrupted by notification review after {waited}s")
                return {"status": "interrupted", "reason": "notification_review", "waited": waited, "user_message": summary}

            due = self.notification_store.get_due_notification()
            if due:
                self.notification_store.record_firing(due["id"])
                waited = int(time.monotonic() - start)
                info(f"[NOTIF] Due notification fired: {due['id']}")
                return {"status": "interrupted", "reason": "notification_due", "waited": waited, "user_message": f'[Notification] id={due["id"]} — Time to show: "{due["message"]}"\nAfter showing it (or deferring), call schedule_notification with this ID to set when it fires next.'}

            if ENABLE_DISPLAY:
                result = http_get("/buttons/state", timeout=2)
                if result.get("button"):
                    http_post("/buttons/reset", {}, timeout=5)
                    self.presence.touch()
                    waited = int(time.monotonic() - start)

                    # In pomodoro mode a press means "I finished a cycle" — log it.
                    if self.pomodoro_mode:
                        self._log_pomodoro_cycle()
                        stats = self.pomodoro_store.stats()
                        info(f"[POMODORO] Button logged a cycle — today {stats['today']}")
                        return {
                            "status": "interrupted", "reason": "pomodoro_button", "waited": waited,
                            "button": result["button"],
                            "user_message": (
                                f"Austin pressed the button — logged a completed pomodoro. "
                                f"Today: {stats['today']}, current streak: {stats['current_streak']} days. "
                                "Give him a quick bit of encouragement."
                            ),
                        }

                    if self.notification_store.has_pending_proposal():
                        approved = self.notification_store.approve_pending()
                        info(f"[NOTIF] Proposal approved: {approved['id']}")
                        user_msg = f'The user approved your notification: "{approved["message"]}"'
                    else:
                        user_msg = "The user pressed a button — they want you to say something!"

                    info(f"[WAIT] Interrupted by button press after {waited}s")
                    return {
                        "status": "interrupted",
                        "reason": f"Button {result['button']} pressed — injected nudge",
                        "button": result["button"],
                        "waited": waited,
                        "user_message": user_msg,
                    }
            time.sleep(BUTTON_CHECK_INTERVAL)

        info(f"[WAIT] Completed {seconds}s.")
        return {"status": "ok", "waited": seconds}

    def _tool_log_drink(self, args: dict) -> dict:
        try:
            mg = int(args.get("mg", 0))
        except (TypeError, ValueError):
            return {"status": "error", "message": "mg must be an integer"}
        label = str(args.get("label", "")).strip()
        if not label:
            return {"status": "error", "message": "No label provided"}
        if mg <= 0:
            return {"status": "error", "message": "mg must be a positive integer"}
        if mg > 1000:
            return {"status": "error", "message": "mg looks too high for one drink — double-check the dose"}
        try:
            minutes_ago = max(0, int(args.get("minutes_ago") or 0))
        except (TypeError, ValueError):
            minutes_ago = 0

        entry = self.drink_store.add(mg, label, minutes_ago)
        self.status_publisher.trigger()
        at_str = time.strftime("%-I:%M%p", time.localtime(entry["t"] / 1000)).lower().lstrip("0")
        total_24h = self.drink_store.total_last_24h_mg()
        info(f"[CAFFEINE] Logged {mg}mg ({label}) at {at_str} — 24h total {total_24h}mg")
        published = "Feed updates within a minute." if self.status_publisher.enabled else "Note: AWS publishing is not configured, logged locally only."
        return {
            "status": "ok",
            "message": f"Logged {mg}mg ({label}) at {at_str}. Last 24h total: {total_24h}mg. {published}",
        }

    def _tool_list_drinks(self, args: dict) -> dict:
        drinks = self.drink_store.list_recent()
        if not drinks:
            return {"status": "ok", "message": "No drinks logged yet.", "drinks": []}
        formatted = []
        for d in drinks:
            t = d.get("t", 0)
            ts_str = time.strftime("%-I:%M%p %a %b %d", time.localtime(t / 1000)).lower().lstrip("0")
            formatted.append({
                "timestamp_ms": t,
                "time": ts_str,
                "mg": d.get("mg", 0),
                "label": d.get("label", ""),
            })
        return {"status": "ok", "drinks": formatted}

    def _tool_edit_drink(self, args: dict) -> dict:
        timestamp_ms = int(args.get("timestamp_ms", 0))
        if not timestamp_ms:
            return {"status": "error", "message": "timestamp_ms is required"}
        mg = args.get("mg")
        if mg is not None:
            try:
                mg = int(mg)
            except (TypeError, ValueError):
                return {"status": "error", "message": "mg must be an integer"}
        label = args.get("label")
        if label is not None:
            label = str(label).strip()
        if mg is None and label is None:
            return {"status": "error", "message": "Provide at least one of mg or label to update"}

        updated = self.drink_store.edit(timestamp_ms, mg=mg, label=label)
        if not updated:
            return {"status": "error", "message": f"No drink found with timestamp_ms={timestamp_ms}. Try list_drinks to find the right timestamp."}
        self.status_publisher.trigger()
        return {
            "status": "ok",
            "message": f"Updated drink: {updated['mg']}mg {updated['label']}",
            "drink": updated,
        }

    # --- Activity ---------------------------------------------------------

    @staticmethod
    def _format_activity_segment(segment: dict, now: float) -> dict:
        started = float(segment.get("started_at", 0))
        ended = segment.get("ended_at")
        end_value = float(ended) if ended is not None else now
        return {
            "presence": segment.get("presence"),
            "activity": segment.get("activity"),
            "started_at": time.strftime(
                "%Y-%m-%d %-I:%M%p", time.localtime(started)
            ).lower().lstrip("0"),
            "ended_at": (
                time.strftime("%Y-%m-%d %-I:%M%p", time.localtime(end_value))
                .lower().lstrip("0")
                if ended is not None else None
            ),
            "duration_seconds": max(0, int(end_value - started)),
            "last_observed_at": segment.get("last_observed_at"),
            "source": segment.get("source", ""),
            "evidence": segment.get("evidence", ""),
            "current": ended is None,
        }

    def _tool_list_activity(self, args: dict) -> dict:
        if not self.activity_store:
            return {"status": "error", "message": "Activity logging is disabled."}
        try:
            limit = max(1, min(int(args.get("limit") or 20), 100))
        except (TypeError, ValueError):
            return {"status": "error", "message": "limit must be an integer."}
        since_hours = args.get("since_hours")
        if since_hours is not None:
            try:
                since_hours = max(0.0, float(since_hours))
            except (TypeError, ValueError):
                return {"status": "error", "message": "since_hours must be a number."}
        now = time.time()
        segments = self.activity_store.list_recent(
            limit=limit,
            since_hours=since_hours,
            now=now,
        )
        formatted = [
            self._format_activity_segment(segment, now) for segment in segments
        ]
        return {
            "status": "ok",
            "current": formatted[-1] if formatted and formatted[-1]["current"] else None,
            "segments": formatted,
        }

    # --- Pomodoro ---------------------------------------------------------

    def _publish_pomodoro_stats(self):
        """MOCK feed publish. The real dev-site feed (public S3, like caffeine)
        is deferred — for now write the payload to a local pomodoro.json so it's
        inspectable and the wiring is ready. Swap this for an S3 put later."""
        try:
            stats = self.pomodoro_store.stats()
            path = os.path.join(PROJECT_DIR, "pomodoro.json")
            with open(path, "w") as f:
                json.dump(stats, f)
            info(f"[POMODORO] (mock feed) wrote stats: {stats}")
        except Exception as e:
            info(f"[POMODORO] Mock feed write failed: {e}")

    def _log_pomodoro_cycle(self, label: str = "pomodoro", minutes_ago: int = 0) -> dict:
        """Append a cycle and refresh mock feed + focus screen. Shared by the
        log_pomodoro tool and the in-mode button press."""
        entry = self.pomodoro_store.add(label, minutes_ago)
        self._publish_pomodoro_stats()
        if self.pomodoro_mode:
            # A finished cycle starts a fresh work block; bump the shown end time.
            self.pomodoro_block_end = time.time() + self.pomodoro_work_minutes * 60
            self.pomodoro_last_activity = time.time()
            self.pomodoro_idle_notified = False
            self._render_pomodoro_screen()
        return entry

    def _note_pomodoro_activity(self):
        """Mark recent user activity so the idle-exit nudge doesn't fire on someone
        who's actively working (chat/motion during a focus session)."""
        if self.pomodoro_mode:
            self.pomodoro_last_activity = time.time()
            self.pomodoro_idle_notified = False

    def _render_pomodoro_screen(self):
        """Draw the focus screen on the e-ink (count + estimated block end)."""
        if not ENABLE_DISPLAY:
            return
        count = self.pomodoro_store.count_today()
        ends = time.strftime("%-I:%M%p", time.localtime(self.pomodoro_block_end)).lower().lstrip("0")
        elapsed = time.monotonic() - self.last_display_time
        if elapsed < MIN_DISPLAY_INTERVAL:
            time.sleep(MIN_DISPLAY_INTERVAL - elapsed)
        # Prefer the dedicated big-number screen; fall back to plain text if the
        # display server predates the /display/pomodoro endpoint.
        ok = http_post("/display/pomodoro", {"count": count, "ends": ends}, timeout=15)
        if not ok:
            http_post("/display", {"text": f"POMODORO MODE\nDone today: {count}\nBlock ends {ends}"}, timeout=15)
        self.last_display_time = time.monotonic()

    def _tool_log_pomodoro(self, args: dict) -> dict:
        label = str(args.get("label") or "pomodoro").strip() or "pomodoro"
        try:
            minutes_ago = max(0, int(args.get("minutes_ago") or 0))
        except (TypeError, ValueError):
            minutes_ago = 0
        entry = self._log_pomodoro_cycle(label, minutes_ago)
        at_str = time.strftime("%-I:%M%p", time.localtime(entry["t"] / 1000)).lower().lstrip("0")
        stats = self.pomodoro_store.stats()
        info(f"[POMODORO] Logged cycle ({label}) at {at_str} — today {stats['today']}, streak {stats['current_streak']}")
        return {
            "status": "ok",
            "message": f"Logged a pomodoro ({label}) at {at_str}. Today: {stats['today']}, current streak: {stats['current_streak']} days.",
            "stats": stats,
        }

    def _tool_list_pomodoros(self, args: dict) -> dict:
        cycles = self.pomodoro_store.list_recent()
        if not cycles:
            return {"status": "ok", "message": "No pomodoros logged yet.", "cycles": []}
        formatted = []
        for c in cycles:
            t = c.get("t", 0)
            ts_str = time.strftime("%-I:%M%p %a %b %d", time.localtime(t / 1000)).lower().lstrip("0")
            formatted.append({"timestamp_ms": t, "time": ts_str, "label": c.get("label", "")})
        return {"status": "ok", "cycles": formatted}

    def _tool_edit_pomodoro(self, args: dict) -> dict:
        timestamp_ms = int(args.get("timestamp_ms", 0))
        if not timestamp_ms:
            return {"status": "error", "message": "timestamp_ms is required"}
        if args.get("delete"):
            if self.pomodoro_store.delete(timestamp_ms):
                self._publish_pomodoro_stats()
                if self.pomodoro_mode:
                    self._render_pomodoro_screen()
                return {"status": "ok", "message": "Deleted that pomodoro cycle."}
            return {"status": "error", "message": f"No cycle found with timestamp_ms={timestamp_ms}. Try list_pomodoros."}
        label = args.get("label")
        if label is not None:
            label = str(label).strip()
        if label is None:
            return {"status": "error", "message": "Provide a label to update, or delete=true to remove."}
        updated = self.pomodoro_store.edit(timestamp_ms, label=label)
        if not updated:
            return {"status": "error", "message": f"No cycle found with timestamp_ms={timestamp_ms}. Try list_pomodoros."}
        self._publish_pomodoro_stats()
        return {"status": "ok", "message": f"Updated cycle label to '{updated['label']}'.", "cycle": updated}

    def _tool_pomodoro_stats(self, args: dict) -> dict:
        stats = self.pomodoro_store.stats()
        return {"status": "ok", "stats": stats}

    def _tool_enter_pomodoro_mode(self, args: dict) -> dict:
        try:
            wm = int(args.get("work_minutes") or POMODORO_WORK_MINUTES)
        except (TypeError, ValueError):
            wm = POMODORO_WORK_MINUTES
        self.pomodoro_work_minutes = max(1, wm)
        self.pomodoro_mode = True
        self.pomodoro_block_end = time.time() + self.pomodoro_work_minutes * 60
        self.pomodoro_last_activity = time.time()
        self.pomodoro_idle_notified = False
        self.presence.touch()
        count = self.pomodoro_store.count_today()
        ends = time.strftime("%-I:%M%p", time.localtime(self.pomodoro_block_end)).lower().lstrip("0")
        self._render_pomodoro_screen()
        info(f"[POMODORO] Entered pomodoro mode ({self.pomodoro_work_minutes}min blocks), {count} done today")
        if ENABLE_DISPLAY:
            btn = "Press the button when you finish a cycle and I'll log it."
        else:
            btn = "Tell me when you finish a cycle and I'll log it."
        return {
            "status": "ok",
            "message": f"Pomodoro mode on. {count} done today, this block ends {ends}. {btn}",
        }

    def _tool_exit_pomodoro_mode(self, args: dict) -> dict:
        was_on = self.pomodoro_mode
        self.pomodoro_mode = False
        self.pomodoro_idle_notified = False
        stats = self.pomodoro_store.stats()
        if ENABLE_DISPLAY and was_on:
            # Return the e-ink to a normal message.
            self._tool_update_display(
                {"text": f"Focus session done. {stats['today']} cycles today. Nice work."},
                speak=False,
            )
        info(f"[POMODORO] Exited pomodoro mode — {stats['today']} today")
        return {
            "status": "ok",
            "message": f"Left pomodoro mode. {stats['today']} cycles today, streak {stats['current_streak']} days.",
            "stats": stats,
        }

    def _tool_propose_notification(self, args: dict) -> dict:
        message = args.get("message", "")
        trigger_type = args.get("trigger_type", "interval")
        trigger_value = args.get("trigger_value", "")

        if not message.strip():
            return {"status": "error", "message": "No message provided"}

        if len(message) > 100:
            return {"status": "error", "message": "Message too long (max 100 chars)"}

        # Reject any existing pending proposal so the new one can replace it
        if self.notification_store.has_pending_proposal():
            self.notification_store.reject_pending()

        notif = self.notification_store.create_proposal(
            message, trigger_type, trigger_value
        )
        info(f"[NOTIF] Proposal created: {notif['id']} — \"{message}\"")
        if ENABLE_DISPLAY:
            response_instructions = (
                f"Show this proposal to the user: '{message}'. Tell them they can "
                "press the physical button or reply in chat."
            )
        else:
            response_instructions = (
                f"Show this proposal to the user: '{message}'. Ask them to reply "
                "naturally with what they want."
            )
        if NOTIFICATION_APPROVAL_MODE == "smart":
            response_instructions += (
                " When they reply in chat, interpret their intent and call "
                "resolve_notification_proposal only if it is clear."
            )
        elif not ENABLE_DISPLAY:
            response_instructions += " The legacy harness recognizes standard yes/no replies."
        return {
            "status": "ok",
            "notification_id": notif["id"],
            "message": f"Proposal saved. {response_instructions}",
        }

    def _tool_resolve_notification_proposal(self, args: dict) -> dict:
        if NOTIFICATION_APPROVAL_MODE != "smart":
            return {
                "status": "error",
                "message": "Notification decisions are handled by the legacy harness in this mode.",
            }

        decision = args.get("decision", "")
        reason = args.get("reason", "")
        if decision not in {"approve", "reject"}:
            return {"status": "error", "message": "Decision must be approve or reject."}
        if not isinstance(reason, str) or not reason.strip():
            return {"status": "error", "message": "A brief decision reason is required."}
        reason = reason.strip()

        pending = self.notification_store.get_pending_proposal()
        if not pending:
            return {"status": "error", "message": "There is no pending notification proposal."}
        if self.last_chat_message_time <= pending.get("proposed_at", 0):
            return {
                "status": "error",
                "message": "Wait for a new user chat response before resolving this proposal.",
            }

        if decision == "approve":
            resolved = self.notification_store.approve_pending()
        else:
            resolved = self.notification_store.reject_pending()
        if not resolved:
            return {"status": "error", "message": "The pending proposal could not be resolved."}

        past_tense = "approved" if decision == "approve" else "rejected"
        info(
            f"[NOTIF] Proposal {past_tense} by smart approval: {resolved['id']} "
            f"(reason: {reason})"
        )
        return {
            "status": "ok",
            "decision": decision,
            "notification_id": resolved["id"],
            "message": f"Notification proposal {past_tense}.",
        }

    def _build_turn_reminder(self) -> str:
        reminder = build_policy_reminder()
        if NOTIFICATION_APPROVAL_MODE != "smart":
            return reminder

        pending = self.notification_store.get_pending_proposal()
        if not pending:
            return reminder
        return (
            f"{reminder}\n\n"
            "PENDING NOTIFICATION PROPOSAL:\n"
            f"- id: {pending['id']}\n"
            f"- message: {pending['message']}\n"
            f"- schedule: {pending['trigger_type']} {pending['trigger_value']}\n"
            "Interpret only a genuine user response sent after this proposal. If it clearly "
            "approves or rejects the proposal, call resolve_notification_proposal. If it is "
            "ambiguous or unrelated, leave the proposal pending."
        )

    def _tool_schedule_notification(self, args: dict) -> dict:
        notif_id = args.get("notification_id", "")
        seconds = args.get("seconds", 600)
        if not notif_id:
            return {"status": "error", "message": "No notification_id provided"}
        seconds = max(10, int(seconds))
        result = self.notification_store.schedule(notif_id, seconds)
        if result:
            info(f"[NOTIF] Scheduled {notif_id} to fire in {seconds}s")
            return {"status": "ok", "message": f"Scheduled to fire again in {seconds}s ({seconds//60}min)."}
        return {"status": "error", "message": f"Notification {notif_id} not found"}

    def _tool_delete_notification(self, args: dict) -> dict:
        notif_id = args.get("notification_id", "")
        if not notif_id:
            return {"status": "error", "message": "No notification_id provided"}
        self.notification_store.delete(notif_id)
        info(f"[NOTIF] Deleted {notif_id}")
        return {"status": "ok", "message": f"Notification {notif_id} deleted."}

    def _detect_patterns(self) -> str | None:
        patterns = []
        now = time.time()

        with self.ctx_lock:
            messages = self.ctx.messages

        take_photo_times = []
        user_event_times = []

        for m in messages:
            ts = m.get("_ts", 0)
            role = m.get("role", "")

            if role == "assistant":
                for tc in m.get("tool_calls", []):
                    if tc.get("function", {}).get("name") == "take_photo":
                        take_photo_times.append(ts)
            elif role == "user":
                user_event_times.append(ts)

        recent_photos = [t for t in take_photo_times if now - t < 5400]
        if len(recent_photos) >= 3:
            recent_chat = [t for t in user_event_times if recent_photos[0] <= t <= recent_photos[-1]]
            if not recent_chat:
                hours = (recent_photos[-1] - recent_photos[0]) / 3600
                patterns.append(f"user at desk for {hours:.0f}+ hours")

        local_hour = time.localtime().tm_hour
        if local_hour >= 23 or local_hour < 6:
            now_str = time.strftime("%-I:%M%p").lower().lstrip("0")
            patterns.append(f"it's late ({now_str})")

        if user_event_times:
            last_event = max(user_event_times)
        else:
            last_event = 0
        if now - last_event > 14400 and 8 <= local_hour < 22:
            hours = (now - last_event) / 3600
            patterns.append(f"no user interaction for {hours:.0f}+ hours")

        return ", ".join(patterns) if patterns else None


    def _idle_wait(self):
        self._set_agent_state("sleeping")
        try:
            # Falling into idle is itself a wait boundary, even if a post
            # arrived just before the idle loop began.
            self._sweep_chat_queue()
            return self._idle_wait_impl()
        finally:
            self._set_agent_state("thinking")

    def _idle_wait_impl(self):
        for _ in range(IDLE_TIMEOUT):
            if not self.running:
                return
            if self.chat_event.is_set():
                self._note_pomodoro_activity()
                self._sweep_chat_queue()
                return
            if self.motion_event.is_set():
                self.motion_event.clear()
                self._note_pomodoro_activity()
                desc = self.motion_description or "Something moved"
                with self.ctx_lock:
                    self.ctx.add_user(f"Motion detected in the room! Here's what the camera sees: {desc}")
                info("[IDLE] Interrupted by motion")
                return
            if ENABLE_DISPLAY:
                result = http_get("/buttons/state", timeout=2)
                if result.get("button"):
                    http_post("/buttons/reset", {}, timeout=5)
                    self.presence.touch()

                    # Pomodoro mode: a press logs a completed cycle.
                    if self.pomodoro_mode:
                        self._log_pomodoro_cycle()
                        stats = self.pomodoro_store.stats()
                        with self.ctx_lock:
                            self.ctx.add_user(
                                f"Austin pressed the button — logged a completed pomodoro. "
                                f"Today: {stats['today']}, current streak: {stats['current_streak']} days. "
                                "Give him a quick bit of encouragement."
                            )
                        info(f"[POMODORO] Button logged a cycle (idle) — today {stats['today']}")
                        return

                    if self.notification_store.has_pending_proposal():
                        approved = self.notification_store.approve_pending()
                        with self.ctx_lock:
                            self.ctx.add_user(
                                f'The user approved your notification: "{approved["message"]}"'
                            )
                        info(f"[NOTIF] Proposal approved in idle: {approved['id']}")
                    else:
                        with self.ctx_lock:
                            self.ctx.add_user("The user pressed a button — they want you to say something!")

                    info("[IDLE] Interrupted by button press")
                    return
            # Preserve the one-second hardware polling cadence while allowing a
            # chat POST to wake an idle agent immediately.
            self.chat_event.wait(1)
        with self.ctx_lock:
            if ENABLE_CAMERA:
                self.ctx.add_user(
                    "Some time has passed. Use take_photo to see the room, or wait to stay quiet."
                )
            else:
                self.ctx.add_user(
                    "Some time has passed. Find something to talk about, or wait to stay quiet."
                )

    def _start_chat_server(self):
        ChatHandler.orchestrator = self
        ChatHandler.session_token = self.session_token
        ChatHandler.use_https = CHAT_USE_HTTPS
        try:
            ChatHandler.asset_version = _chat_asset_version(ChatHandler.static_dir)
        except OSError as error:
            info(f"[CHAT] Could not hash static assets: {error}")
            ChatHandler.asset_version = "missing"

        ssl_ctx = None
        if CHAT_USE_HTTPS:
            ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ssl_ctx.load_cert_chain(SSL_CERT_FILE, SSL_KEY_FILE)

        class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
            daemon_threads = True

            def get_request(self):
                conn, addr = self.socket.accept()
                if ssl_ctx is None:
                    return conn, addr
                # Don't blindly wrap in TLS: peek the first byte so a plaintext
                # http:// request gets a helpful "use https://" reply instead of a
                # bare connection reset. A TLS ClientHello always starts with the
                # handshake record type 0x16; an HTTP request starts with an ASCII
                # method (GET/POST/...). Leave non-TLS conns raw so ChatHandler can
                # detect them and redirect.
                try:
                    conn.settimeout(5)
                    first = conn.recv(1, socket.MSG_PEEK)
                    conn.settimeout(None)
                except OSError:
                    conn.close()
                    raise
                if first[:1] == b"\x16":
                    try:
                        conn = ssl_ctx.wrap_socket(conn, server_side=True)
                    except OSError:
                        conn.close()
                        raise
                return conn, addr

        server = ThreadedHTTPServer(("0.0.0.0", CHAT_SERVER_PORT), ChatHandler)
        if CHAT_USE_HTTPS:
            info(f"[CHAT] HTTPS server listening on :{CHAT_SERVER_PORT}")
        else:
            info(f"[CHAT] Server listening on :{CHAT_SERVER_PORT}")
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()

    def cleanup(self):
        info("Cleaning up...")
        tts_interrupt()
        self.status_publisher.stop()
        with self.ctx_lock:
            self.ctx.save()
        try:
            self.notification_store._save()
        except Exception:
            pass
        try:
            if self.camera:
                self.camera.close()
        except Exception:
            pass
        info("Done.")


LOGIN_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Friend — Login</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,sans-serif;background:#1a1a2e;color:#e0e0e0;height:100vh;display:flex;align-items:center;justify-content:center}
form{background:#16213e;padding:32px;border-radius:12px;display:flex;flex-direction:column;gap:16px;width:320px;max-width:90vw}
h1{font-size:20px;text-align:center}
input{padding:12px;border:1px solid #0f3460;border-radius:8px;background:#1a1a2e;color:#e0e0e0;font-size:16px;outline:none}
input:focus{border-color:#e94560}
button{padding:12px;background:#e94560;color:#fff;border:none;border-radius:8px;font-size:16px;cursor:pointer}
button:hover{background:#c73e54}
.error{color:#e94560;font-size:14px;text-align:center;display:none}
</style></head><body>
<form id="form" method="post" action="/login">
<h1>AI Friend</h1>
<input type="password" id="password" name="password" placeholder="Password" autocomplete="current-password" autofocus>
<button type="submit">Login</button>
<div class="error" id="error">Wrong password</div>
</form>
<script>
const params=new URLSearchParams(window.location.search);
if(params.get('e')==='1')document.getElementById('error').style.display='block';
</script>
</body></html>"""

NUDGE_PREFIXES = [
    "Some time has passed",
    "Display is updated. Continue the rhythm",
    "You just woke up!",
    "Here is the latest photo",
    "The user pressed a button",
    "[Notification review]",
    "[Notification]",
    "The user approved your notification:",
    "Motion detected in the room!",
    "Austin pressed the button",
    "You've been in pomodoro mode",
]


class ChatHandler(BaseHTTPRequestHandler):
    orchestrator = None
    session_token = None
    use_https = False
    asset_version = "dev"
    static_dir = os.path.join(PROJECT_DIR, "static")

    def log_message(self, format, *args):
        pass  # suppress default access logs

    def _get_cookie(self, name):
        cookie_header = self.headers.get("Cookie", "")
        for cookie in cookie_header.split(";"):
            cookie = cookie.strip()
            if "=" in cookie:
                k, v = cookie.split("=", 1)
                if k.strip() == name:
                    return v.strip()
        return None

    def _check_auth(self):
        return self._get_cookie("session") == self.session_token

    def _require_auth(self):
        if self._check_auth():
            return True
        if self.path.startswith("/chat") or self.path.startswith("/static/"):
            self.send_error(401)
        else:
            self._send_login()
        return False

    def _send_login(self):
        data = LOGIN_HTML.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _set_auth_cookie(self):
        max_age = CHAT_SESSION_DAYS * 86400
        secure = "; Secure" if self.use_https else ""
        self.send_header("Set-Cookie", f"session={self.session_token}; Path=/; Max-Age={max_age}; HttpOnly; SameSite=Lax{secure}")

    def _redirect_to_https(self):
        """Client reached the TLS port over plaintext http://. Reply with a
        readable message and a redirect so browsers auto-upgrade."""
        host = self.headers.get("Host") or f"localhost:{CHAT_SERVER_PORT}"
        location = f"https://{host}{self.path}"
        body = (
            f"This server requires HTTPS. Use {location}\n"
        ).encode()
        self.send_response(307)  # Temporary Redirect — preserves method, not cached
        self.send_header("Location", location)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _plaintext_on_tls(self):
        """True when the server expects TLS but this connection is plaintext."""
        return self.use_https and not isinstance(self.connection, ssl.SSLSocket)

    def do_GET(self):
        if self._plaintext_on_tls():
            return self._redirect_to_https()
        path = self.path.split("?", 1)[0]
        if path == "/login":
            self._send_login()
        elif path == "/":
            if not self._require_auth():
                return
            self._serve_html()
        elif path == "/chat":
            if not self._require_auth():
                return
            self._get_messages()
        elif path == "/chat/events":
            if not self._require_auth():
                return
            self._serve_events()
        elif path.startswith("/chat/media/"):
            if not self._require_auth():
                return
            self._get_media()
        elif path in (
            "/static/chat.css",
            "/static/chat.mjs",
            "/static/chat_model.mjs",
        ):
            if not self._require_auth():
                return
            self._serve_static(path)
        else:
            self.send_error(404)

    def do_POST(self):
        if self._plaintext_on_tls():
            return self._redirect_to_https()
        path = self.path.split("?", 1)[0]
        if path == "/login":
            self._handle_login()
        elif path == "/chat":
            if not self._require_auth():
                return
            self._post_message()
        else:
            self.send_error(404)

    def do_PATCH(self):
        if self._plaintext_on_tls():
            return self._redirect_to_https()
        path = self.path.split("?", 1)[0]
        if not path.startswith("/chat/queue/"):
            self.send_error(404)
            return
        if not self._require_auth():
            return
        self._patch_queued_message(path.removeprefix("/chat/queue/"))

    def do_DELETE(self):
        if self._plaintext_on_tls():
            return self._redirect_to_https()
        path = self.path.split("?", 1)[0]
        if not path.startswith("/chat/queue/"):
            self.send_error(404)
            return
        if not self._require_auth():
            return
        self._delete_queued_message(path.removeprefix("/chat/queue/"))

    def _handle_login(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode()
        password = ""
        for pair in body.split("&"):
            if "=" in pair:
                k, v = pair.split("=", 1)
                if k == "password":
                    from urllib.parse import unquote
                    password = unquote(v.strip())
        if password == CHAT_PASSWORD:
            info("[AUTH] Login succeeded")
            self.send_response(302)
            self._set_auth_cookie()
            self.send_header("Location", "/")
            self.end_headers()
        else:
            info("[AUTH] Failed login attempt")
            self.send_response(302)
            self.send_header("Location", "/login?e=1")
            self.end_headers()

    def _serve_html(self):
        path = os.path.join(self.static_dir, "chat.html")
        try:
            with open(path, "r", encoding="utf-8") as source:
                html = source.read()
        except OSError:
            self.send_error(500)
            return
        html = (
            html
            .replace("__CHAT_MAX_IMAGES__", str(CHAT_MAX_IMAGES_PER_MESSAGE))
            .replace("__CHAT_MAX_MEDIA_BYTES__", str(CHAT_MAX_MEDIA_BYTES))
            .replace(
                "__LLM_SUPPORTS_IMAGES__",
                json.dumps(LLM_SUPPORTS_IMAGES),
            )
            .replace("__CHAT_TAKEOVER_SECONDS__", str(CHAT_TAKEOVER_SECONDS))
            .replace("__CHAT_ASSET_VERSION__", self.asset_version)
        )
        data = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_static(self, path: str):
        filenames = {
            "/static/chat.css": ("chat.css", "text/css; charset=utf-8"),
            "/static/chat.mjs": ("chat.mjs", "text/javascript; charset=utf-8"),
            "/static/chat_model.mjs": ("chat_model.mjs", "text/javascript; charset=utf-8"),
        }
        filename, content_type = filenames[path]
        try:
            with open(os.path.join(self.static_dir, filename), "rb") as source:
                data = source.read()
        except OSError:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "private, max-age=31536000, immutable")
        self.send_header("ETag", f'"{self.asset_version}"')
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: dict, status: int = 200):
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> dict | None:
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            self._send_json({"error": "Invalid Content-Length"}, 400)
            return None
        if length <= 0:
            self._send_json({"error": "Empty request"}, 400)
            return None
        if length > CHAT_MAX_REQUEST_BYTES:
            self.close_connection = True
            self._send_json({"error": "Request is too large."}, 413)
            return None
        try:
            body = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json({"error": "Invalid JSON"}, 400)
            return None
        if not isinstance(body, dict):
            self._send_json({"error": "Request body must be a JSON object"}, 400)
            return None
        return body

    def _get_media(self):
        media_id = self.path.removeprefix("/chat/media/").split("?", 1)[0]
        if len(media_id) != 64 or any(c not in "0123456789abcdef" for c in media_id):
            self.send_error(404)
            return

        found = ChatHandler.orchestrator._find_chat_media(media_id)
        if not found:
            self.send_error(404)
            return

        mime, raw = found
        self.send_response(200)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "private, max-age=3600")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(raw)

    def _get_messages(self):
        orch = ChatHandler.orchestrator
        msgs, queued = orch._snapshot_chat_sources()
        context_bubbles = []
        for m in msgs:
            role = m.get("role")
            content = m.get("content", "")
            ts = m.get("_ts")
            ts_str = orch._chat_time(ts)
            chat_id = m.get("_chat_id")
            if not chat_id:
                continue
            if role == "user":
                if isinstance(content, list):
                    model_text = "\n".join(
                        p.get("text", "")
                        for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    ).strip()
                    text = str(m.get("_chat_original_text", model_text))
                    images = [
                        {
                            "url": f"/chat/media/{a['id']}",
                            "name": a.get("name", "image"),
                            "type": a.get("type", ""),
                        }
                        for a in m.get("_chat_images", [])
                        if isinstance(a, dict) and a.get("id")
                    ]
                    if text or images:
                        context_bubbles.append({
                            "id": f"{chat_id}:user",
                            "role": role,
                            "content": text,
                            "images": images,
                            "time": ts_str,
                            "queued": False,
                        })
                else:
                    if not content or not content.strip():
                        continue
                    display_content = str(m.get("_chat_original_text", content))
                    if not display_content.strip():
                        continue
                    if any(display_content.startswith(p) for p in NUDGE_PREFIXES):
                        continue
                    context_bubbles.append({
                        "id": f"{chat_id}:user",
                        "role": role,
                        "content": display_content,
                        "time": ts_str,
                        "images": [],
                        "queued": False,
                    })
            elif role == "assistant":
                # Show display updates and chat messages
                for tool_index, tc in enumerate(m.get("tool_calls", [])):
                    fn_name = tc.get("function", {}).get("name")
                    if fn_name in ("update_display", "send_chat_message"):
                        try:
                            args = json.loads(tc["function"]["arguments"])
                            msg_text = args.get("text", "")
                            if msg_text.strip():
                                context_bubbles.append({
                                    "id": f"{chat_id}:assistant:{tool_index}",
                                    "role": "assistant",
                                    "content": msg_text,
                                    "images": [],
                                    "time": ts_str,
                                    "queued": False,
                                })
                        except (json.JSONDecodeError, KeyError):
                            pass

        # Server contract: context bubbles always precede queue bubbles,
        # regardless of timestamp. Never sort this combined list by time.
        queue_bubbles = [orch._queue_bubble(entry) for entry in queued]
        filtered = (context_bubbles + queue_bubbles)[-50:]
        ui_snapshot = orch.ui_state.snapshot()
        self._send_json({
            "messages": filtered,
            "agent": ui_snapshot["agent"],
            "activity": ui_snapshot["activity"],
            "server_id": ui_snapshot["server_id"],
            "chat_revision": ui_snapshot["chat_revision"],
        })

    def _post_message(self):
        body = self._read_json()
        if body is None:
            return
        raw_message = body.get("message", "")
        if not isinstance(raw_message, str):
            self._send_json({"error": "message must be a string"}, 400)
            return
        message = raw_message.strip()
        if body.get("images") and not LLM_SUPPORTS_IMAGES:
            self._send_json(
                {"error": "The configured brain model does not support image uploads."},
                400,
            )
            return
        try:
            content, chat_images = build_chat_message(
                message,
                body.get("images", []),
            )
        except ChatMediaError as e:
            self._send_json({"error": str(e)}, e.status_code)
            return

        orch = ChatHandler.orchestrator
        entry = {
            "id": f"q_{uuid.uuid4().hex}",
            "created_at": time.time(),
            "text": message,
            "content": content,
            "chat_images": chat_images,
        }
        with orch.chat_queue_lock:
            orch.chat_queue.append(entry)
            orch.chat_event.set()
        orch.last_chat_message_time = time.time()
        orch.presence.touch()
        orch.ui_state.chat_changed()
        media_log = f", {len(chat_images)} attachment(s)" if chat_images else ""
        info(f"[CHAT] User message: {message[:100] or '[media only]'}{media_log}")

        self._send_json({
            "status": "ok",
            "id": entry["id"],
            "message": orch._queue_bubble(entry),
        })

    @staticmethod
    def _valid_queue_id(queue_id: str) -> bool:
        return (
            len(queue_id) == 34
            and queue_id.startswith("q_")
            and all(char in "0123456789abcdef" for char in queue_id[2:])
        )

    def _delete_queued_message(self, queue_id: str):
        if not self._valid_queue_id(queue_id):
            self.send_error(404)
            return
        restored = ChatHandler.orchestrator._undo_queued_message(queue_id)
        if restored is None:
            self._send_json(
                {"error": "That message was already sent and cannot be undone."},
                409,
            )
            return
        self._send_json({"status": "ok", "restored": restored})

    def _patch_queued_message(self, queue_id: str):
        if not self._valid_queue_id(queue_id):
            self.send_error(404)
            return
        body = self._read_json()
        if body is None:
            return
        message = body.get("message")
        if not isinstance(message, str):
            self._send_json({"error": "message must be a string"}, 400)
            return
        try:
            updated = ChatHandler.orchestrator._edit_queued_message(
                queue_id, message
            )
        except ChatMediaError as error:
            self._send_json({"error": str(error)}, error.status_code)
            return
        if updated is None:
            self._send_json(
                {"error": "That message was already sent and cannot be edited."},
                409,
            )
            return
        self._send_json({"status": "ok", "message": updated})

    def _serve_events(self):
        orch = ChatHandler.orchestrator
        if not orch.sse_slots.acquire(blocking=False):
            self._send_json({"error": "Too many event streams."}, 503)
            return

        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(b"retry: 2000\n\n")

            snapshot = orch.ui_state.snapshot()
            revision = snapshot["event_revision"]
            last_change = time.monotonic()

            def write_snapshot(value):
                payload = json.dumps({
                    "agent": value["agent"],
                    "activity": value["activity"],
                    "server_id": value["server_id"],
                    "chat_revision": value["chat_revision"],
                }, separators=(",", ":")).encode()
                self.wfile.write(b"event: snapshot\n")
                self.wfile.write(b"data: " + payload + b"\n\n")
                self.wfile.flush()

            write_snapshot(snapshot)
            while orch.running:
                idle_remaining = CHAT_SSE_IDLE_SECONDS - (
                    time.monotonic() - last_change
                )
                if idle_remaining <= 0:
                    break
                timeout = min(CHAT_SSE_HEARTBEAT_SECONDS, idle_remaining)
                snapshot, changed = orch.ui_state.wait_after(revision, timeout)
                if changed:
                    revision = snapshot["event_revision"]
                    last_change = time.monotonic()
                    write_snapshot(snapshot)
                else:
                    self.wfile.write(b": heartbeat\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            self.close_connection = True
            orch.sse_slots.release()


def main():
    logger.VERBOSE_LOG = "/home/austingibb/ai_desk_agent/verbose.log"
    orch = Orchestrator()
    try:
        orch.run()
    finally:
        orch.cleanup()


if __name__ == "__main__":
    main()
