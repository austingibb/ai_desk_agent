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
    REVIEW_INTERVAL,
    POLICY_REMINDER,
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
)
from notifications import NotificationStore
from caffeine import DrinkStore
from presence import ActiveTracker
from status_publisher import StatusPublisher
from context import Context
from ai_client import AIClient, LLMError, VisionClient
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
        self.chat_queue = []       # queued chat messages (added by handler, drained by main loop)
        self.chat_queue_lock = threading.Lock()
        self.ctx_lock = threading.Lock()
        self.mcp_tools = []
        self.mcp = None
        self.notification_store = NotificationStore()
        self.drink_store = DrinkStore()
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
        info("Init AI client (DeepSeek on OpenRouter)...")
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
                        self.ctx.add_user(msg)

            with self.ctx_lock:
                self.ctx._repair_pairing()
                messages = self.ctx.get_messages()
                msg_tokens = self.ctx.total_tokens()
            messages.append({"role": "user", "content": POLICY_REMINDER})
            estimated = msg_tokens + estimate_tool_tokens(tools) + len(POLICY_REMINDER) // 4
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
                messages.append({"role": "user", "content": POLICY_REMINDER})
                estimated = msg_tokens + estimate_tool_tokens(tools) + len(POLICY_REMINDER) // 4
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

            if response["reasoning"]:
                info(f"[REASONING] {response['reasoning'][:200]}...")
                info(f"[REASONING] {response['reasoning']}")
            if response["content"]:
                info(f"[AI] {response['content'][:200]}")
                info(f"[AI] {response['content']}")

            if not response["tool_calls"]:
                # If DeepSeek returned text but no tool call, display it automatically
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
                waited = int(time.monotonic() - start)
                info(f"[WAIT] Interrupted by chat message after {waited}s")
                return {"status": "interrupted", "reason": "chat_message", "waited": waited}

            if self.motion_event.is_set():
                self.motion_event.clear()
                waited = int(time.monotonic() - start)
                desc = self.motion_description or "Something moved"
                info(f"[WAIT] Interrupted by motion after {waited}s")
                return {"status": "interrupted", "reason": "motion_detected", "waited": waited,
                        "user_message": f"Motion detected in the room! Here's what the camera sees: {desc}"}

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
                return
            if self.motion_event.is_set():
                self.motion_event.clear()
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
#form{display:flex;gap:8px;padding:12px;background:#16213e;border-top:1px solid #0f3460}
#input{flex:1;padding:10px 14px;border:1px solid #0f3460;border-radius:20px;background:#1a1a2e;color:#e0e0e0;font-size:15px;outline:none}
#input:focus{border-color:#e94560}
#input:disabled{opacity:0.4;cursor:not-allowed}
button{padding:10px 20px;background:#e94560;color:#fff;border:none;border-radius:20px;font-size:15px;cursor:pointer}
button:hover{background:#c73e54}
button:disabled{opacity:0.4;cursor:not-allowed}
</style></head><body>
<div id="status-banner"></div>
<div id="messages"></div>
<form id="form"><input id="input" placeholder="Say something..." autocomplete="off"><button type="submit">Send</button></form>
<script>
const div=document.getElementById('messages');
const banner=document.getElementById('status-banner');
const form=document.getElementById('form');
const input=document.getElementById('input');
const rendered=new Set();
let initialized=false;

function msgKey(m){
  return m.role+'|'+m.content.slice(0,80);
}

function msgHTML(m){
  return `<div class="msg-wrap ${m.role}"><div class="role">${m.role}${m.time?' · '+m.time:''}</div><div class="msg">${m.content.replace(/</g,'&lt;')}</div></div>`;
}

function atBottom(){
  return div.scrollHeight-div.scrollTop-div.clientHeight<60;
}

function updateStatus(status){
  if(status){
    banner.textContent=status;
    banner.classList.add('show');
    input.disabled=true;
    form.querySelector('button').disabled=true;
  }else{
    banner.classList.remove('show');
    input.disabled=false;
    form.querySelector('button').disabled=false;
  }
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
form.onsubmit=async e=>{
  e.preventDefault();
  const msg=input.value.trim();
  if(!msg)return;
  input.value='';
  const resp=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:msg})});
  if(resp.ok){
    setTimeout(refresh,500);
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
        if self.path == "/chat":
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
        if self.path.startswith("/login"):
            self._send_login()
        elif self.path == "/":
            if not self._require_auth():
                return
            self._serve_html()
        elif self.path == "/chat":
            if not self._require_auth():
                return
            self._get_messages()
        else:
            self.send_error(404)

    def do_POST(self):
        if self._plaintext_on_tls():
            return self._redirect_to_https()
        if self.path == "/login":
            self._handle_login()
        elif self.path == "/chat":
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
        data = CHAT_HTML.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

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
                if qm.strip() not in ctx_user_contents:
                    msgs.append({"role": "user", "content": qm, "_ts": time.time()})

        filtered = []
        for m in msgs:
            role = m.get("role")
            content = m.get("content", "")
            ts = m.get("_ts")
            ts_str = time.strftime("%-I:%M%p %a", time.localtime(ts)).lower().lstrip("0") if ts else ""
            if role == "user":
                if isinstance(content, list):
                    continue  # image messages
                if not content or not content.strip():
                    continue
                if any(content.startswith(p) for p in NUDGE_PREFIXES):
                    continue
                filtered.append({"role": role, "content": content, "time": ts_str})
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
        data = json.dumps({"messages": filtered, "status": status}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _post_message(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        message = body.get("message", "").strip()

        if not message:
            self.send_error(400, "Empty message")
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
            orch.chat_queue.append(message)
        orch.chat_event.set()
        orch.presence.touch()
        info(f"[CHAT] User message: {message[:100]}")

        data = json.dumps({"status": "ok"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    logger.VERBOSE_LOG = "/home/austingibb/ai_desk_agent/verbose.log"
    orch = Orchestrator()
    try:
        orch.run()
    finally:
        orch.cleanup()


if __name__ == "__main__":
    main()
