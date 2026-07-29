#!/usr/bin/env python3
"""AI E-Ink Friend — agent loop orchestrator. Runs on Pi 5 (192.168.0.39)."""

import time
import os
import signal
import sys
import json
import secrets
import socket
import ssl
import threading
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
    REVIEW_INTERVAL,
    build_policy_reminder,
    estimate_tool_tokens,
    LLM_ESTIMATED_MAX_TOKENS,
    COMPACT_AFTER_N_MESSAGES,
    ENABLE_CAMERA,
    ENABLE_DISPLAY,
    VISION_POLL_INTERVAL,
    MOTION_POLL_INTERVAL,
    CHILL_TIMEOUT,
    VISION_REQUESTS_FILE,
    SCENE_RMS_THRESHOLD,
    SCENE_PCT_THRESHOLD,
    SCENE_MAX_STALE_SECONDS,
    ENABLE_REOLINK,
    REOLINK_IP,
    REOLINK_USER,
    REOLINK_PASSWORD,
    REOLINK_TIMEOUT,
    POMODORO_WORK_MINUTES,
    POMODORO_IDLE_EXIT_SECONDS,
)
from notifications import NotificationStore
from caffeine import DrinkStore
from pomodoro import PomodoroStore
from presence import ActiveTracker
from status_publisher import StatusPublisher
from context import Context
from ai_client import AIClient, LLMError, VisionClient
from chat_media import ChatMediaError, build_chat_message, media_data_from_message
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


class Orchestrator:
    def __init__(self):
        self.ctx = Context()
        # Camera pulls in picamera2 (Pi-only) — import it lazily so the agent
        # still runs on a plain laptop with ENABLE_CAMERA=0.
        if ENABLE_CAMERA:
            from camera import Camera
            self.camera = Camera()
        else:
            self.camera = None
        self.ai = AIClient()
        self.vision = VisionClient() if ENABLE_CAMERA else None
        self.reolink = ReoLinkCamera(REOLINK_IP, REOLINK_USER, REOLINK_PASSWORD, REOLINK_TIMEOUT) if ENABLE_REOLINK else None
        self.running = True
        self.last_display_time = 0
        self.chat_event = threading.Event()
        # Entries are either legacy text strings or {"content": multipart,
        # "chat_images": UI metadata} for user-uploaded images/GIFs.
        self.chat_queue = []
        self.chat_queue_lock = threading.Lock()
        self.ctx_lock = threading.Lock()
        self.mcp_tools = []
        self.mcp = None
        self.notification_store = NotificationStore()
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
        self.latest_scene = None  # {"description": str, "timestamp": float}
        self.scene_lock = threading.Lock()
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

        # Status tracking for chat UI
        self.status_message = ""
        self.status_lock = threading.Lock()

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

        try:
            info("Init MCP client...")
            self.mcp = MCPClient()
            tools = self.mcp.initialize()
            self.mcp_tools = self.mcp.get_tool_definitions()
            info(f"[MCP] Discovered {len(tools)} tools: {[t['name'] for t in tools]}")
        except Exception as e:
            info(f"[MCP] Unavailable: {e}")

        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        info("\nShutting down...")
        self.running = False

    def run(self):
        info("Init camera...")
        info(f"Init AI client ({self.ai.model} on OpenRouter)...")
        info("Init vision client (local Gemma)...")
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
                    self.ctx.messages.insert(0, {"role": "system", "content": prompt, "_ts": self.ctx._now()})
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
        info("Entering agent loop.")

        while self.running:
            try:
                self._turn()
            except LLMError as e:
                info(f"[FATAL] Unhandled LLM error: {e}")
                self._llm_failures += 1
                delay = self._llm_backoff_seconds()
                info(f"[FATAL] Backing off for {delay}s")
                time.sleep(delay)
            except Exception as e:
                info(f"[ERROR] {e}")
                time.sleep(5)

        self.cleanup()

    def _turn(self):
        tool_call_count = 0
        last_tool_name = None

        while self.running:
            tools = list(get_tool_definitions())
            if self.mcp_tools:
                tools.extend(self.mcp_tools)

            # Drain queued chat messages into context at a safe point
            with self.chat_queue_lock:
                queued = list(self.chat_queue)
                self.chat_queue.clear()
            if queued:
                with self.ctx_lock:
                    for msg in queued:
                        if isinstance(msg, dict):
                            self.ctx.add_user(
                                msg.get("content", ""),
                                chat_images=msg.get("chat_images"),
                            )
                        else:
                            self.ctx.add_user(msg)

            with self.ctx_lock:
                self.ctx.demote_old_images()
                self.ctx._repair_pairing()
                messages = self.ctx.get_messages()
                msg_tokens = self.ctx.total_tokens()
            reminder = build_policy_reminder()
            messages.append({"role": "user", "content": reminder})
            estimated = msg_tokens + estimate_tool_tokens(tools) + len(reminder) // 4
            if estimated > LLM_ESTIMATED_MAX_TOKENS:
                info(f"[LLM] Token estimate {estimated} exceeds limit {LLM_ESTIMATED_MAX_TOKENS}, compacting...")
                with self.status_lock:
                    self.status_message = "Compacting memory..."
                try:
                    self.ctx.check_compact(self.ai, self.ctx_lock)
                except LLMError as e:
                    info(f"[LLM] Compaction failed during token overflow: {e}")
                finally:
                    with self.status_lock:
                        self.status_message = ""
                with self.ctx_lock:
                    messages = self.ctx.get_messages()
                    msg_tokens = self.ctx.total_tokens()
                reminder = build_policy_reminder()
                messages.append({"role": "user", "content": reminder})
                estimated = msg_tokens + estimate_tool_tokens(tools) + len(reminder) // 4
                info(f"[LLM] After compaction: ~{msg_tokens} msg tokens + {estimate_tool_tokens(tools)} tool tokens = ~{estimated} total")
            info(f"[LLM] Sending {len(messages)} messages (~{msg_tokens} msg tokens, ~{estimate_tool_tokens(tools)} tool tokens, ~{estimated} total)...")
            play_sound("thinking")
            try:
                response = self.ai.chat_with_tools(messages, tools)
            except LLMError as e:
                recoverable = e.status_code >= 500 or e.status_code == 429
                err_str = str(e)
                if "exceed_context_size_error" in err_str or "exceeds the available context size" in err_str:
                    info(f"[LLM] Context overflow detected, triggering compaction...")
                    try:
                        self.ctx.check_compact(self.ai, self.ctx_lock)
                    except LLMError as ce:
                        info(f"[LLM] Compaction failed during overflow: {ce}")
                    time.sleep(1)
                    continue
                if not recoverable:
                    info(f"[LLM] Non-recoverable error (HTTP {e.status_code}), backing off: {e}")
                    self._llm_failures += 1
                    self._last_llm_fail = time.time()
                    delay = self._llm_backoff_seconds()
                    info(f"[LLM] Backing off for {delay}s ({self._llm_failures} consecutive failures)")
                    self._display_error(f"LLM API error ({e.status_code}). Retrying in {delay // 60}m...")
                    time.sleep(delay)
                    continue
                else:
                    info(f"[LLM] Recoverable error (HTTP {e.status_code}), backing off: {e}")
                    self._llm_failures += 1
                    self._last_llm_fail = time.time()
                    delay = min(self._llm_backoff_seconds(), 120)  # cap transient retries at 2min
                    info(f"[LLM] Backing off for {delay}s ({self._llm_failures} consecutive failures)")
                    time.sleep(delay)
                    continue
            except (requests.Timeout, requests.ConnectionError) as e:
                info(f"[LLM] Network error: {e}")
                self._llm_failures += 1
                self._last_llm_fail = time.time()
                delay = min(self._llm_backoff_seconds(), 120)
                info(f"[LLM] Backing off for {delay}s")
                time.sleep(delay)
                continue
            except Exception as e:
                info(f"[LLM] Unexpected error: {e}")
                self._llm_failures += 1
                self._last_llm_fail = time.time()
                time.sleep(5)
                continue

            self._llm_failures = 0

            with self.ctx_lock:
                self.ctx.add_assistant(response)
                self.ctx.note_latest_image_response(
                    _visible_response_text(response)
                )

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
                        result = self._tool_send_chat_message({"text": content})
                    else:
                        info(f"[AUTO-DISPLAY] AI returned content without update_display, showing it...")
                        result = self._tool_update_display({"text": content})
                    if result.get("status") == "ok":
                        self._tool_wait({})
                    continue
                info("[IDLE] AI produced no tool calls. Waiting...")
                self._idle_wait()
                return

            # Execute all tool calls, deferring user messages until after
            # all tool results are added (OpenRouter requires tool results
            # to immediately follow the assistant message, no interleaving)
            deferred_user_msgs = []

            for tc in response["tool_calls"]:
                tool_call_count += 1
                last_tool_name = tc["name"]
                info(f"[TOOL] {tc['name']}({tc['arguments']})")
                try:
                    result = self._execute_tool(tc["name"], tc["arguments"])
                except Exception as e:
                    result = {"status": "error", "message": f"Tool execution failed: {e}"}
                    info(f"[TOOL ERROR] {e}")
                info(f"[TOOL RESULT] {json.dumps(result)[:200]}")
                info(f"[TOOL RESULT] {json.dumps(result)}")
                with self.ctx_lock:
                    self.ctx.add_tool_result(tc["id"], tc["name"], result)
                user_msg = result.get("user_message")
                if user_msg:
                    deferred_user_msgs.append(user_msg)

            # Now add deferred user messages (after all tool results)
            if deferred_user_msgs:
                with self.ctx_lock:
                    for msg in deferred_user_msgs:
                        self.ctx.add_user(msg)

            try:
                with self.ctx_lock:
                    self.ctx.demote_old_images()
                    will_compact = len(self.ctx.messages) >= COMPACT_AFTER_N_MESSAGES
                if will_compact:
                    with self.status_lock:
                        self.status_message = "Compacting memory..."
                try:
                    self.ctx.check_compact(self.ai, self.ctx_lock)
                finally:
                    if will_compact:
                        with self.status_lock:
                            self.status_message = ""
                # Only merge summaries when user is away (chill mode) to avoid
                # blocking the agent loop with back-to-back LLM calls
                if self.vision_mode == "chill" or not ENABLE_CAMERA:
                    with self.status_lock:
                        self.status_message = "Merging memory..."
                    try:
                        self.ctx.check_merge_summaries(self.ai, self.ctx_lock)
                    finally:
                        with self.status_lock:
                            self.status_message = ""
            except LLMError as e:
                info(f"[LLM] Compaction failed: {e}")

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
        return {
            "status": "ok",
            "description": scene["description"],
            "captured_at": captured_at,
            "age_seconds": age,
        }

    def _tool_capture_photo(self) -> dict:
        """Take a photo now and block until the vision model describes it."""
        info("[PHOTO] Synchronous capture + describe (blocking, may take up to 120s)...")
        scene = self._capture_and_describe()
        if not scene:
            return {"status": "error", "message": "Failed to capture or describe photo — vision model may be unavailable"}
        captured_at = time.strftime("%-I:%M%p", time.localtime(scene["timestamp"])).lower().lstrip("0")
        return {
            "status": "ok",
            "description": scene["description"],
            "captured_at": captured_at,
        }

    def _tool_update_vision_requests(self, args: dict) -> dict:
        requests_text = args.get("requests", "").strip()
        if not requests_text:
            return {"status": "error", "message": "No requests text provided"}

        # Read current contents so the AI can see what's already there
        current = ""
        try:
            with open(VISION_REQUESTS_FILE, "r") as f:
                current = f.read().strip()
        except FileNotFoundError:
            pass

        # First call with existing content: bounce back so the AI can merge
        if current and not self.vision_requests_shown:
            self.vision_requests_shown = True
            info(f"[VISION] Bouncing update_vision_requests — showing existing requests first")
            return {
                "status": "needs_retry",
                "message": (
                    "STOP — the vision requests file already has content. "
                    "Review the existing requests below and call update_vision_requests again "
                    "with your new requests MERGED with the existing ones. "
                    "Don't drop existing requests unless they're truly no longer needed."
                ),
                "current_requests": current,
            }

        try:
            with open(VISION_REQUESTS_FILE, "w") as f:
                f.write(f"# Requests for Image Model\n\n{requests_text}\n")
            self.vision_requests_shown = False  # reset so next update bounces again
            info(f"[VISION] Requests updated: {requests_text[:100]}...")
            return {"status": "ok", "message": "Vision requests updated. Changes take effect on the next photo capture."}
        except Exception as e:
            return {"status": "error", "message": f"Failed to write requests file: {e}"}

    def _tool_take_reolink_photo(self) -> dict:
        if not self.reolink:
            return {"status": "error", "message": "Reolink camera not initialized"}
        if not self.vision:
            return {"status": "error", "message": "Vision model not available (ENABLE_CAMERA=0)"}
        info("[REOLINK] Capturing snapshot...")
        try:
            _, data_uri = self.reolink.capture()
        except Exception as e:
            return {"status": "error", "message": f"Reolink capture failed: {e}"}
        try:
            description = self.vision.describe(data_uri)
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

    def _capture_and_describe(self) -> dict | None:
        """Capture a photo and get a text description from the local vision model."""
        try:
            jpeg_bytes, photo_uri = self.camera.capture()
        except Exception as e:
            info(f"[VISION] Camera error: {e}")
            return None

        self._save_debug_image(jpeg_bytes)

        try:
            description = self.vision.describe(photo_uri)
        except Exception as e:
            info(f"[VISION] Describe error: {e}")
            return None
        if not description:
            info("[VISION] Got empty description from vision model, skipping")
            return None
        scene = {"description": description, "timestamp": time.time()}
        with self.scene_lock:
            self.latest_scene = scene
        return scene

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
        try:
            jpeg_bytes, photo_uri = self.camera.capture()
        except Exception as e:
            info(f"[VISION] Full capture error: {e}")
            return

        self._save_debug_image(jpeg_bytes)

        try:
            description = self.vision.describe(photo_uri)
        except Exception as e:
            info(f"[VISION] Describe error: {e}")
            return

        if not description:
            info("[VISION] Empty description, skipping")
            return

        scene = {"description": description, "timestamp": time.time()}
        with self.scene_lock:
            self.latest_scene = scene
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
                self.chat_event.clear()
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
        return {
            "status": "ok",
            "message": f"Proposal saved. Now show it to the user with update_display: '{message} — press button to approve!'",
        }

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
        for _ in range(IDLE_TIMEOUT):
            if not self.running:
                return
            if self.chat_event.is_set():
                self.chat_event.clear()
                self._note_pomodoro_activity()
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
            time.sleep(1)
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


CHAT_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Friend Chat</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,sans-serif;background:#1a1a2e;color:#e0e0e0;height:100vh;display:flex;flex-direction:column}
#status-banner{display:none;background:#e94560;color:#fff;text-align:center;padding:8px;font-size:13px;font-weight:600}
#status-banner.show{display:block}
#messages{flex:1;overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:12px}
.msg-wrap{display:flex;flex-direction:column;max-width:80%}
.msg-wrap.user{align-self:flex-end}
.msg-wrap.assistant{align-self:flex-start}
.role{font-size:11px;opacity:0.6;margin-bottom:3px;padding:0 4px}
.msg-wrap.user .role{text-align:right}
.msg{padding:10px 14px;border-radius:12px;word-wrap:break-word;line-height:1.4}
.msg-wrap.user .msg{background:#0f3460;color:#e0e0e0}
.msg-wrap.assistant .msg{background:#16213e;color:#e0e0e0}
.media-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px;margin-top:8px}
.media-grid:empty{display:none}
.chat-media{display:block;max-width:100%;max-height:360px;border-radius:8px;object-fit:contain;background:#0b1224}
#composer{background:#16213e;border-top:1px solid #0f3460}
#attachments{display:none;gap:8px;padding:10px 12px 0;overflow-x:auto}
#attachments.show{display:flex}
.attachment{position:relative;flex:0 0 80px;height:64px}
.attachment img{width:80px;height:64px;object-fit:cover;border-radius:8px;border:1px solid #0f3460}
.attachment button{position:absolute;right:-5px;top:-7px;width:22px;height:22px;padding:0;border-radius:50%;font-size:14px;line-height:22px}
#upload-error{display:none;color:#ff8ca0;font-size:12px;padding:7px 14px 0}
#upload-error.show{display:block}
#form{display:flex;gap:8px;padding:12px;background:#16213e;border-top:1px solid #0f3460}
#input{flex:1;padding:10px 14px;border:1px solid #0f3460;border-radius:20px;background:#1a1a2e;color:#e0e0e0;font-size:15px;outline:none}
#input:focus{border-color:#e94560}
#input:disabled{opacity:0.4;cursor:not-allowed}
button{padding:10px 20px;background:#e94560;color:#fff;border:none;border-radius:20px;font-size:15px;cursor:pointer}
button:hover{background:#c73e54}
button:disabled{opacity:0.4;cursor:not-allowed}
#attach{padding:10px 14px;background:#0f3460}
#attach:hover{background:#174b82}
body.dragging:after{content:'Drop images or GIFs to attach';position:fixed;inset:12px;display:flex;align-items:center;justify-content:center;border:3px dashed #e94560;border-radius:16px;background:rgba(26,26,46,.92);color:#fff;font-size:20px;font-weight:600;z-index:10;pointer-events:none}
</style></head><body>
<div id="status-banner"></div>
<div id="messages"></div>
<div id="composer">
<div id="upload-error"></div>
<div id="attachments"></div>
<form id="form">
<input id="file-input" type="file" accept="image/png,image/jpeg,image/webp,image/gif" multiple hidden>
<button id="attach" type="button" title="Attach images or GIFs">Image</button>
<input id="input" placeholder="Say something..." autocomplete="off">
<button id="send" type="submit">Send</button>
</form>
</div>
<script>
const div=document.getElementById('messages');
const banner=document.getElementById('status-banner');
const form=document.getElementById('form');
const input=document.getElementById('input');
const attach=document.getElementById('attach');
const fileInput=document.getElementById('file-input');
const attachmentDiv=document.getElementById('attachments');
const uploadError=document.getElementById('upload-error');
const rendered=new Set();
const allowedTypes=new Set(['image/png','image/jpeg','image/webp','image/gif']);
const maxImages=__CHAT_MAX_IMAGES__;
const maxMediaBytes=__CHAT_MAX_MEDIA_BYTES__;
let selectedFiles=[];
let initialized=false;
let sending=false;

function escapeHTML(value){
  return String(value??'').replace(/[&<>"']/g,ch=>({
    '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'
  })[ch]);
}

function msgKey(m){
  const images=(m.images||[]).map(i=>i.url).join(',');
  return m.role+'|'+(m.content||'').slice(0,80)+'|'+images+'|'+(m.time||'');
}

function msgHTML(m){
  const text=escapeHTML(m.content||'').replace(/\\n/g,'<br>');
  const images=(m.images||[]).map(i=>
    `<a href="${escapeHTML(i.url)}" target="_blank" rel="noopener"><img class="chat-media" src="${escapeHTML(i.url)}" alt="${escapeHTML(i.name||'Uploaded image')}" loading="lazy"></a>`
  ).join('');
  return `<div class="msg-wrap ${escapeHTML(m.role)}"><div class="role">${escapeHTML(m.role)}${m.time?' · '+escapeHTML(m.time):''}</div><div class="msg">${text}<div class="media-grid">${images}</div></div></div>`;
}

function atBottom(){
  return div.scrollHeight-div.scrollTop-div.clientHeight<60;
}

function updateStatus(status){
  if(status){
    banner.textContent=status;
    banner.classList.add('show');
    input.disabled=true;
    form.querySelectorAll('button').forEach(button=>button.disabled=true);
  }else{
    banner.classList.remove('show');
    input.disabled=false;
    if(!sending)form.querySelectorAll('button').forEach(button=>button.disabled=false);
  }
}

function showUploadError(message=''){
  uploadError.textContent=message;
  uploadError.classList.toggle('show',Boolean(message));
}

function renderAttachments(){
  attachmentDiv.innerHTML=selectedFiles.map((item,index)=>
    `<div class="attachment"><img src="${item.url}" alt="${escapeHTML(item.file.name)}"><button type="button" data-remove="${index}" title="Remove attachment">x</button></div>`
  ).join('');
  attachmentDiv.classList.toggle('show',selectedFiles.length>0);
}

function clearAttachments(){
  selectedFiles.forEach(item=>URL.revokeObjectURL(item.url));
  selectedFiles=[];
  fileInput.value='';
  renderAttachments();
}

function addFiles(files){
  showUploadError();
  for(const file of files){
    if(!allowedTypes.has(file.type)){
      showUploadError(`${file.name} is not a supported image or GIF.`);
      continue;
    }
    if(selectedFiles.length>=maxImages){
      showUploadError(`You can attach up to ${maxImages} files.`);
      break;
    }
    const duplicate=selectedFiles.some(item=>
      item.file.name===file.name&&item.file.size===file.size&&item.file.lastModified===file.lastModified
    );
    const total=selectedFiles.reduce((sum,item)=>sum+item.file.size,0);
    if(!duplicate&&total+file.size>maxMediaBytes){
      showUploadError(`Attachments can total at most ${Math.floor(maxMediaBytes/1024/1024)} MB.`);
      continue;
    }
    if(!duplicate)selectedFiles.push({file,url:URL.createObjectURL(file)});
  }
  renderAttachments();
}

function filePayload(item){
  return new Promise((resolve,reject)=>{
    const reader=new FileReader();
    reader.onload=()=>resolve({
      name:item.file.name,
      type:item.file.type,
      data_url:reader.result
    });
    reader.onerror=()=>reject(new Error(`Could not read ${item.file.name}.`));
    reader.readAsDataURL(item.file);
  });
}

async function refresh(){
  try{
    const r=await fetch('/chat');
    const data=await r.json();
    updateStatus(data.status||'');
    const msgs=data.messages||[];
    if(!initialized){
      div.innerHTML=msgs.map(msgHTML).join('');
      msgs.forEach(m=>rendered.add(msgKey(m)));
      initialized=true;
      div.scrollTop=div.scrollHeight;
      return;
    }
    const wasAtBottom=atBottom();
    let added=false;
    for(const m of msgs){
      const key=msgKey(m);
      if(!rendered.has(key)){
        div.insertAdjacentHTML('beforeend',msgHTML(m));
        rendered.add(key);
        added=true;
      }
    }
    if(added&&wasAtBottom)div.scrollTop=div.scrollHeight;
  }catch(e){}
}
setInterval(refresh,2000);
refresh();
attach.onclick=()=>fileInput.click();
fileInput.onchange=()=>addFiles(fileInput.files);
attachmentDiv.onclick=e=>{
  const button=e.target.closest('[data-remove]');
  if(!button)return;
  const index=Number(button.dataset.remove);
  const removed=selectedFiles.splice(index,1)[0];
  if(removed)URL.revokeObjectURL(removed.url);
  renderAttachments();
};
let dragDepth=0;
document.addEventListener('dragenter',e=>{
  if(Array.from(e.dataTransfer?.items||[]).some(item=>item.kind==='file')){
    dragDepth++;
    document.body.classList.add('dragging');
  }
});
document.addEventListener('dragleave',()=>{
  dragDepth=Math.max(0,dragDepth-1);
  if(!dragDepth)document.body.classList.remove('dragging');
});
document.addEventListener('dragover',e=>e.preventDefault());
document.addEventListener('drop',e=>{
  e.preventDefault();
  dragDepth=0;
  document.body.classList.remove('dragging');
  addFiles(e.dataTransfer?.files||[]);
});
document.addEventListener('paste',e=>{
  const files=Array.from(e.clipboardData?.files||[]).filter(file=>allowedTypes.has(file.type));
  if(files.length){
    e.preventDefault();
    addFiles(files);
  }
});
form.onsubmit=async e=>{
  e.preventDefault();
  const msg=input.value.trim();
  if((!msg&&!selectedFiles.length)||sending)return;
  sending=true;
  showUploadError();
  input.disabled=true;
  form.querySelectorAll('button').forEach(button=>button.disabled=true);
  try{
    const images=await Promise.all(selectedFiles.map(filePayload));
    const resp=await fetch('/chat',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({message:msg,images})
    });
    let data={};
    try{data=await resp.json();}catch(e){}
    if(!resp.ok)throw new Error(data.error||`Upload failed (${resp.status}).`);
    input.value='';
    clearAttachments();
    setTimeout(refresh,500);
  }catch(error){
    showUploadError(error.message||'Could not send that message.');
  }finally{
    sending=false;
    if(!banner.classList.contains('show')){
      input.disabled=false;
      form.querySelectorAll('button').forEach(button=>button.disabled=false);
      input.focus();
    }
  }
};
</script></body></html>"""

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
        if self.path.startswith("/chat"):
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
        elif path.startswith("/chat/media/"):
            if not self._require_auth():
                return
            self._get_media()
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
        html = (
            CHAT_HTML
            .replace("__CHAT_MAX_IMAGES__", str(CHAT_MAX_IMAGES_PER_MESSAGE))
            .replace("__CHAT_MAX_MEDIA_BYTES__", str(CHAT_MAX_MEDIA_BYTES))
        )
        data = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
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

    def _get_media(self):
        media_id = self.path.removeprefix("/chat/media/").split("?", 1)[0]
        if len(media_id) != 64 or any(c not in "0123456789abcdef" for c in media_id):
            self.send_error(404)
            return

        orch = ChatHandler.orchestrator
        found = None
        with orch.ctx_lock:
            for msg in reversed(orch.ctx.messages):
                found = media_data_from_message(msg, media_id)
                if found:
                    break
        if not found:
            # The message may still be queued while the main loop is busy.
            with orch.chat_queue_lock:
                queued = list(orch.chat_queue)
            for entry in reversed(queued):
                if not isinstance(entry, dict):
                    continue
                queued_msg = {
                    "content": entry.get("content", []),
                    "_chat_images": entry.get("chat_images", []),
                }
                found = media_data_from_message(queued_msg, media_id)
                if found:
                    break
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
        with orch.ctx_lock:
            msgs = list(orch.ctx.messages)  # raw messages with _ts, no timestamp injection
        # Include queued messages that haven't been drained to context yet,
        # but skip any that already appear in ctx (dedupe by content)
        ctx_user_contents = set()
        for m in msgs:
            if m.get("role") == "user":
                c = m.get("content", "")
                if isinstance(c, str) and c.strip():
                    ctx_user_contents.add(c.strip())
        with orch.chat_queue_lock:
            for qm in orch.chat_queue:
                if isinstance(qm, dict):
                    msgs.append({
                        "role": "user",
                        "content": qm.get("content", ""),
                        "_chat_images": qm.get("chat_images", []),
                        "_ts": time.time(),
                    })
                elif qm.strip() not in ctx_user_contents:
                    msgs.append({"role": "user", "content": qm, "_ts": time.time()})

        filtered = []
        for m in msgs:
            role = m.get("role")
            content = m.get("content", "")
            ts = m.get("_ts")
            ts_str = time.strftime("%-I:%M%p %a", time.localtime(ts)).lower().lstrip("0") if ts else ""
            if role == "user":
                if isinstance(content, list):
                    text = " ".join(
                        p.get("text", "")
                        for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    ).strip()
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
                        filtered.append({
                            "role": role,
                            "content": text,
                            "images": images,
                            "time": ts_str,
                        })
                else:
                    if not content or not content.strip():
                        continue
                    display_content = m.get("_chat_original_text", content)
                    if any(display_content.startswith(p) for p in NUDGE_PREFIXES):
                        continue
                    filtered.append({
                        "role": role,
                        "content": display_content,
                        "time": ts_str,
                    })
            elif role == "assistant":
                # Show display updates and chat messages
                for tc in m.get("tool_calls", []):
                    fn_name = tc.get("function", {}).get("name")
                    if fn_name in ("update_display", "send_chat_message"):
                        try:
                            args = json.loads(tc["function"]["arguments"])
                            msg_text = args.get("text", "")
                            if msg_text.strip():
                                filtered.append({"role": "assistant", "content": msg_text, "time": ts_str})
                        except (json.JSONDecodeError, KeyError):
                            pass

        # Return last 50 messages
        filtered = filtered[-50:]
        with orch.status_lock:
            status = orch.status_message
        self._send_json({"messages": filtered, "status": status})

    def _post_message(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            self._send_json({"error": "Invalid Content-Length"}, 400)
            return
        if length <= 0:
            self._send_json({"error": "Empty request"}, 400)
            return
        if length > CHAT_MAX_REQUEST_BYTES:
            self.close_connection = True
            self._send_json(
                {
                    "error": (
                        f"Upload is too large. The attachment limit is "
                        f"{CHAT_MAX_MEDIA_BYTES // (1024 * 1024)} MB."
                    )
                },
                413,
            )
            return
        try:
            body = json.loads(self.rfile.read(length))
        except (json.JSONDecodeError, UnicodeDecodeError):
            self._send_json({"error": "Invalid JSON"}, 400)
            return
        if not isinstance(body, dict):
            self._send_json({"error": "Request body must be a JSON object"}, 400)
            return
        raw_message = body.get("message", "")
        if not isinstance(raw_message, str):
            self._send_json({"error": "message must be a string"}, 400)
            return
        message = raw_message.strip()
        try:
            content, chat_images = build_chat_message(
                message,
                body.get("images", []),
            )
        except ChatMediaError as e:
            self._send_json({"error": str(e)}, e.status_code)
            return

        orch = ChatHandler.orchestrator

        REJECTION_KEYWORDS = ["no", "nah", "don't", "stop", "cancel", "never", "quit", "not that"]
        AFFIRMATION_KEYWORDS = ["yes", "yeah", "yep", "yup", "sure", "ok", "okay", "sounds good", "go for it", "do it", "approve"]

        approval_notice = None
        if orch.notification_store.has_pending_proposal():
            msg_lower = message.lower()
            if any(kw in msg_lower for kw in REJECTION_KEYWORDS):
                rejected = orch.notification_store.reject_pending()
                if rejected:
                    info(f"[NOTIF] Proposal rejected via chat: {rejected['id']}")
            elif not ENABLE_DISPLAY and any(kw in msg_lower for kw in AFFIRMATION_KEYWORDS):
                # No buttons in chat-only mode — an affirmative chat reply approves.
                approved = orch.notification_store.approve_pending()
                if approved:
                    info(f"[NOTIF] Proposal approved via chat: {approved['id']}")
                    approval_notice = f'The user approved your notification: "{approved["message"]}"'

        with orch.chat_queue_lock:
            if approval_notice:
                orch.chat_queue.append(approval_notice)
            if chat_images:
                orch.chat_queue.append({
                    "content": content,
                    "chat_images": chat_images,
                })
            else:
                orch.chat_queue.append(content)
        orch.chat_event.set()
        orch.presence.touch()
        media_log = f", {len(chat_images)} attachment(s)" if chat_images else ""
        info(f"[CHAT] User message: {message[:100] or '[media only]'}{media_log}")

        self._send_json({"status": "ok"})


def main():
    logger.VERBOSE_LOG = "/home/austingibb/ai_desk_agent/verbose.log"
    orch = Orchestrator()
    try:
        orch.run()
    finally:
        orch.cleanup()


if __name__ == "__main__":
    main()
