#!/usr/bin/env python3
"""AI E-Ink Friend — agent loop orchestrator. Runs on Pi 5 (192.168.0.39)."""

import time
import signal
import sys
import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import requests
from config import (
    DISPLAY_SERVER_URL,
    build_system_prompt,
    get_tool_definitions,
    MAX_TOOL_CALLS_PER_TURN,
    MIN_DISPLAY_INTERVAL,
    MAX_WAIT_SECONDS,
    IDLE_TIMEOUT,
    BUTTON_CHECK_INTERVAL,
    CHAT_SERVER_PORT,
    BACKOFF_BASE,
    BACKOFF_MAX,
    REVIEW_INTERVAL,
    MAX_PROPOSAL_INTERVAL,
    CATEGORY_COOLDOWN_REVIEWS,
    POLICY_REMINDER,
    estimate_tool_tokens,
    LLM_ESTIMATED_MAX_TOKENS,
    ENABLE_CAMERA,
    VISION_POLL_INTERVAL,
    VISION_REQUESTS_FILE,
)
from notifications import NotificationStore
from context import Context
from camera import Camera
from ai_client import AIClient, VisionClient
from mcp_client import MCPClient
from sounds import play as play_sound

LOG_FILE = "/home/austingibb/ai_eink/verbose.log"


def log(msg: str):
    try:
        with open(LOG_FILE, "a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def http_get(path: str, timeout: int = 5) -> dict:
    try:
        r = requests.get(f"{DISPLAY_SERVER_URL}{path}", timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"[HTTP GET] {path}: {e}")
    return {}


def http_post(path: str, data: dict, timeout: int = 5):
    try:
        r = requests.post(f"{DISPLAY_SERVER_URL}{path}", json=data, timeout=timeout)
        return r.status_code == 200
    except Exception as e:
        print(f"[HTTP POST] {path}: {e}")
    return False


class Orchestrator:
    def __init__(self):
        self.ctx = Context()
        self.camera = Camera()
        self.ai = AIClient()
        self.vision = VisionClient()
        self.running = True
        self.last_display_time = 0
        self.chat_event = threading.Event()
        self.chat_queue = []       # queued chat messages (added by handler, drained by main loop)
        self.chat_queue_lock = threading.Lock()
        self.ctx_lock = threading.Lock()
        self.backoff = BACKOFF_BASE
        self.mcp_tools = []
        self.mcp = None
        self.notification_store = NotificationStore()
        self.last_review_time = time.time()
        self.last_fired_notification_id = None
        self.last_proposal_time = 0.0
        self.proposal_category_cooldowns = {}

        # Vision background thread state
        self.latest_scene = None  # {"description": str, "timestamp": float}
        self.scene_lock = threading.Lock()

        try:
            print("Init MCP client...")
            self.mcp = MCPClient()
            tools = self.mcp.initialize()
            self.mcp_tools = self.mcp.get_tool_definitions()
            print(f"[MCP] Discovered {len(tools)} tools: {[t['name'] for t in tools]}")
        except Exception as e:
            print(f"[MCP] Unavailable: {e}")

        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        print("\nShutting down...")
        self.running = False

    def _reset_backoff(self):
        self.backoff = BACKOFF_BASE

    def run(self):
        print("Init camera...")
        print("Init AI client (DeepSeek on OpenRouter)...")
        print("Init vision client (local Gemma)...")
        self._start_chat_server()
        if ENABLE_CAMERA:
            self._start_vision_loop()
        with self.ctx_lock:
            if self.ctx.load():
                print("Resuming from saved context.")
                if ENABLE_CAMERA:
                    self.ctx.add_user("You just woke back up after a restart! Use take_photo to see the room and pick up where you left off.")
                else:
                    self.ctx.add_user("You just woke back up after a restart! Camera is not available — use your other tools to pick up where you left off.")
            else:
                self.ctx.add_system(build_system_prompt())
                if ENABLE_CAMERA:
                    self.ctx.add_user("You just woke up! Use take_photo to see the room and say hi.")
                else:
                    self.ctx.add_user("You just woke up! Note: camera/vision tools are not available. Use your other tools to say hi.")
        print("Entering agent loop.")

        while self.running:
            try:
                self._turn()
            except Exception as e:
                print(f"[ERROR] {e}")
                time.sleep(5)

        self.cleanup()

    def _turn(self):
        tool_call_count = 0
        last_tool_name = None
        nudged = False

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
                print(f"[LLM] Token estimate {estimated} exceeds limit {LLM_ESTIMATED_MAX_TOKENS}, compacting...")
                with self.ctx_lock:
                    self.ctx.check_compact(self.ai)
                with self.ctx_lock:
                    messages = self.ctx.get_messages()
                    msg_tokens = self.ctx.total_tokens()
                messages.append({"role": "user", "content": POLICY_REMINDER})
                estimated = msg_tokens + estimate_tool_tokens(tools) + len(POLICY_REMINDER) // 4
                print(f"[LLM] After compaction: ~{msg_tokens} msg tokens + {estimate_tool_tokens(tools)} tool tokens = ~{estimated} total")
            print(f"[LLM] Sending {len(messages)} messages (~{msg_tokens} msg tokens, ~{estimate_tool_tokens(tools)} tool tokens, ~{estimated} total)...")
            play_sound("thinking")
            try:
                response = self.ai.chat_with_tools(messages, tools)
            except Exception as e:
                err_str = str(e)
                if "exceed_context_size_error" in err_str or "exceeds the available context size" in err_str:
                    print(f"[LLM] Context overflow detected, triggering compaction...")
                    with self.ctx_lock:
                        self.ctx.check_compact(self.ai)
                    time.sleep(1)
                    continue
                print(f"[LLM] Error: {e}")
                with self.ctx_lock:
                    self.ctx.add_user("Something went wrong. Continue the rhythm — what's your next action?")
                time.sleep(5)
                continue

            with self.ctx_lock:
                self.ctx.add_assistant(response)

            if response["reasoning"]:
                print(f"[REASONING] {response['reasoning'][:200]}...")
                log(f"[REASONING] {response['reasoning']}")
            if response["content"]:
                print(f"[AI] {response['content'][:200]}")
                log(f"[AI] {response['content']}")

            if not response["tool_calls"]:
                if tool_call_count > 0 and not nudged:
                    print("[PROMPT] No tool call after tool execution, nudging LLM to continue rhythm...")
                    with self.ctx_lock:
                        self.ctx.add_user("Continue the rhythm — what's your next action?")
                    nudged = True
                    continue
                print("[IDLE] AI produced no tool calls. Waiting...")
                self._idle_wait()
                return

            # Execute all tool calls, deferring user messages until after
            # all tool results are added (OpenRouter requires tool results
            # to immediately follow the assistant message, no interleaving)
            deferred_user_msgs = []
            enforced_wait = False
            has_explicit_wait = any(tc["name"] == "wait" for tc in response["tool_calls"])

            for tc in response["tool_calls"]:
                tool_call_count += 1
                last_tool_name = tc["name"]
                print(f"[TOOL] {tc['name']}({tc['arguments']})")
                try:
                    result = self._execute_tool(tc["name"], tc["arguments"])
                except Exception as e:
                    result = {"status": "error", "message": f"Tool execution failed: {e}"}
                    print(f"[TOOL ERROR] {e}")
                print(f"[TOOL RESULT] {json.dumps(result)[:200]}")
                log(f"[TOOL RESULT] {json.dumps(result)}")
                with self.ctx_lock:
                    self.ctx.add_tool_result(tc["id"], tc["name"], result)
                user_msg = result.get("user_message")
                if user_msg:
                    deferred_user_msgs.append(user_msg)

                if tc["name"] == "update_display" and result.get("status") == "ok":
                    enforced_wait = True

            # Enforced wait after display update — skip if DeepSeek already called wait
            if enforced_wait and not has_explicit_wait:
                wait_result = self._tool_wait({})
                print(f"[WAIT ENFORCED] display updated + wait ({wait_result.get('waited', 0)}s)")
                with self.ctx_lock:
                    wait_id = f"wait_{int(time.time())}"
                    # Append wait to the assistant's tool_calls so pairing is valid
                    for m in reversed(self.ctx.messages):
                        if m.get("role") == "assistant" and m.get("tool_calls"):
                            m["tool_calls"].append({
                                "id": wait_id,
                                "type": "function",
                                "function": {"name": "wait", "arguments": "{}"},
                            })
                            break
                    self.ctx.add_tool_result(wait_id, "wait", wait_result)
                wait_user_msg = wait_result.get("user_message")
                if wait_user_msg:
                    deferred_user_msgs.append(wait_user_msg)

            # Now add deferred user messages (after all tool results)
            if deferred_user_msgs:
                with self.ctx_lock:
                    for msg in deferred_user_msgs:
                        self.ctx.add_user(msg)

            with self.ctx_lock:
                self.ctx.check_compact(self.ai)

    def _execute_tool(self, name: str, args: dict) -> dict:
        if name == "take_photo":
            if not ENABLE_CAMERA:
                return {"error": "Camera is disabled. Use other tools instead."}
            play_sound("take_photo")
            return self._tool_take_photo()
        elif name == "update_display":
            play_sound("update_display")
            return self._tool_update_display(args)
        elif name == "wait":
            play_sound("wait")
            return self._tool_wait(args)
        elif name == "update_vision_requests":
            return self._tool_update_vision_requests(args)
        elif name == "propose_notification":
            play_sound("update_display")
            return self._tool_propose_notification(args)
        else:
            if self.mcp:
                play_sound("search")
                try:
                    return self.mcp.call_tool(name, args)
                except Exception as e:
                    return {"error": f"MCP tool '{name}' failed: {e}"}
            return {"error": f"Unknown tool: {name}. Available: take_photo, update_display, wait"}

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

    def _tool_update_vision_requests(self, args: dict) -> dict:
        requests_text = args.get("requests", "").strip()
        if not requests_text:
            return {"status": "error", "message": "No requests text provided"}
        try:
            with open(VISION_REQUESTS_FILE, "w") as f:
                f.write(f"# Requests for Image Model\n\n{requests_text}\n")
            print(f"[VISION] Requests updated: {requests_text[:100]}...")
            return {"status": "ok", "message": "Vision requests updated. Changes take effect on the next photo capture."}
        except Exception as e:
            return {"status": "error", "message": f"Failed to write requests file: {e}"}

    def _capture_and_describe(self) -> dict | None:
        """Capture a photo and get a text description from the local vision model."""
        try:
            jpeg_bytes, photo_uri = self.camera.capture()
        except Exception as e:
            print(f"[VISION] Camera error: {e}")
            return None

        self._save_debug_image(jpeg_bytes)

        try:
            description = self.vision.describe(photo_uri)
        except Exception as e:
            print(f"[VISION] Describe error: {e}")
            return None
        if not description:
            print("[VISION] Got empty description from vision model, skipping")
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
            print(f"[VISION] Failed to save debug image: {e}")
            return
        cutoff = time.time() - 86400
        for old in _glob.glob(_os.path.join(debug_dir, "*.jpg")):
            try:
                if _os.path.getmtime(old) < cutoff:
                    _os.remove(old)
            except Exception:
                pass

    def _start_vision_loop(self):
        def loop():
            print(f"[VISION] Background thread started (interval={VISION_POLL_INTERVAL}s)")
            while self.running:
                scene = self._capture_and_describe()
                if scene:
                    print(f"[VISION] Scene updated: {scene['description'][:100]}...")
                else:
                    print("[VISION] Failed to capture/describe, will retry next interval")
                for _ in range(VISION_POLL_INTERVAL):
                    if not self.running:
                        return
                    time.sleep(1)
            print("[VISION] Background thread stopped")

        t = threading.Thread(target=loop, daemon=True)
        t.start()

    def _tool_update_display(self, args: dict) -> dict:
        text = args.get("text", "")

        if not text.strip():
            return {"status": "error", "message": "No text provided"}

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

    def _tool_wait(self, args: dict) -> dict:
        seconds = max(5, min(MAX_WAIT_SECONDS, self.backoff))
        print(f"[WAIT] Sleeping {seconds}s (backoff={self.backoff}s)...")

        start = time.monotonic()
        while time.monotonic() - start < seconds:
            if self.chat_event.is_set():
                self.chat_event.clear()
                self._reset_backoff()
                waited = int(time.monotonic() - start)
                print(f"[WAIT] Interrupted by chat message after {waited}s")
                return {"status": "interrupted", "reason": "chat_message", "waited": waited}

            if not self.running:
                return {"status": "interrupted", "reason": "shutdown", "waited": int(time.monotonic() - start)}

            if time.time() - self.last_review_time > REVIEW_INTERVAL:
                if self.backoff < 270:
                    self.notification_store.expire_pending()
                    self._tick_cooldowns()
                    patterns = self._detect_patterns()
                    cooldowns = {c: v for c, v in self.proposal_category_cooldowns.items() if v > 0}
                    summary = self.notification_store.get_review_summary(
                        patterns=patterns, cooldown_categories=cooldowns
                    )
                    self.last_review_time = time.time()
                    waited = int(time.monotonic() - start)
                    print(f"[WAIT] Interrupted by notification review after {waited}s")
                    return {"status": "interrupted", "reason": "notification_review", "waited": waited, "user_message": summary}

            due = self.notification_store.get_due_notification()
            if due:
                if self.last_fired_notification_id:
                    self.notification_store.decay_unacknowledged(self.last_fired_notification_id)
                self.notification_store.record_firing(due["id"])
                self.last_fired_notification_id = due["id"]
                self._reset_backoff()
                waited = int(time.monotonic() - start)
                print(f"[NOTIF] Due notification fired: {due['id']}")
                return {"status": "interrupted", "reason": "notification_due", "waited": waited, "user_message": f'[Notification] Time to show: "{due["message"]}"'}

            result = http_get("/buttons/state", timeout=2)
            if result.get("button"):
                self._reset_backoff()
                http_post("/buttons/reset", {}, timeout=5)
                waited = int(time.monotonic() - start)

                if self.notification_store.has_pending_proposal():
                    approved = self.notification_store.approve_pending()
                    print(f"[NOTIF] Proposal approved: {approved['id']}")
                    user_msg = f'The user approved your notification: "{approved["message"]}"'
                else:
                    if self.last_fired_notification_id:
                        self.notification_store.record_acknowledgment(self.last_fired_notification_id)
                        self.last_fired_notification_id = None
                    user_msg = "The user pressed a button — they want you to say something!"

                print(f"[WAIT] Interrupted by button press after {waited}s")
                return {
                    "status": "interrupted",
                    "reason": f"Button {result['button']} pressed — injected nudge",
                    "button": result["button"],
                    "waited": waited,
                    "user_message": user_msg,
                }
            time.sleep(BUTTON_CHECK_INTERVAL)

        self.backoff = min(self.backoff * 3, BACKOFF_MAX)
        print(f"[WAIT] Completed. Next backoff: {self.backoff}s")
        return {"status": "ok", "waited": seconds}

    def _tool_propose_notification(self, args: dict) -> dict:
        message = args.get("message", "")
        category = args.get("category", "misc")
        trigger_type = args.get("trigger_type", "interval")
        trigger_value = args.get("trigger_value", "")

        if not message.strip():
            return {"status": "error", "message": "No message provided"}

        if len(message) > 100:
            return {"status": "error", "message": "Message too long (max 100 chars)"}

        if self.notification_store.has_pending_proposal():
            return {
                "status": "error",
                "message": "A proposal is already pending. Wait for the user to respond first.",
            }

        if time.time() - self.last_proposal_time < MAX_PROPOSAL_INTERVAL:
            remaining = int(MAX_PROPOSAL_INTERVAL - (time.time() - self.last_proposal_time))
            return {
                "status": "error",
                "message": f"Rate limited. Can propose again in {remaining}s.",
            }

        self.last_proposal_time = time.time()
        notif = self.notification_store.create_proposal(
            message, category, trigger_type, trigger_value
        )
        self.proposal_category_cooldowns[category] = CATEGORY_COOLDOWN_REVIEWS
        print(f"[NOTIF] Proposal created: {notif['id']} — \"{message}\"")
        return {
            "status": "ok",
            "message": f"Proposal saved. Now show it to the user with update_display: '{message} — press button to approve!'",
        }

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

    def _tick_cooldowns(self):
        expired = [c for c, v in self.proposal_category_cooldowns.items() if v <= 0]
        for c in expired:
            del self.proposal_category_cooldowns[c]
        for c in list(self.proposal_category_cooldowns.keys()):
            self.proposal_category_cooldowns[c] -= 1

    def _idle_wait(self):
        for _ in range(IDLE_TIMEOUT):
            if not self.running:
                return
            if self.chat_event.is_set():
                self.chat_event.clear()
                self._reset_backoff()
                return
            result = http_get("/buttons/state", timeout=2)
            if result.get("button"):
                self._reset_backoff()
                http_post("/buttons/reset", {}, timeout=5)

                if self.notification_store.has_pending_proposal():
                    approved = self.notification_store.approve_pending()
                    with self.ctx_lock:
                        self.ctx.add_user(
                            f'The user approved your notification: "{approved["message"]}"'
                        )
                    print(f"[NOTIF] Proposal approved in idle: {approved['id']}")
                else:
                    if self.last_fired_notification_id:
                        self.notification_store.record_acknowledgment(self.last_fired_notification_id)
                        self.last_fired_notification_id = None
                    with self.ctx_lock:
                        self.ctx.add_user("The user pressed a button — they want you to say something!")

                print("[IDLE] Interrupted by button press")
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
        server = HTTPServer(("0.0.0.0", CHAT_SERVER_PORT), ChatHandler)
        t = threading.Thread(target=server.serve_forever, daemon=True)
        t.start()
        print(f"[CHAT] Server listening on :{CHAT_SERVER_PORT}")

    def cleanup(self):
        print("Cleaning up...")
        with self.ctx_lock:
            self.ctx.save()
        try:
            self.notification_store._save()
        except Exception:
            pass
        try:
            self.camera.close()
        except Exception:
            pass
        print("Done.")


CHAT_HTML = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Friend Chat</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,sans-serif;background:#1a1a2e;color:#e0e0e0;height:100vh;display:flex;flex-direction:column}
#messages{flex:1;overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:8px}
.msg{padding:10px 14px;border-radius:12px;max-width:80%;word-wrap:break-word;line-height:1.4}
.user{align-self:flex-end;background:#0f3460;color:#e0e0e0}
.assistant{align-self:flex-start;background:#16213e;color:#e0e0e0}
.role{font-size:11px;opacity:0.6;margin-bottom:2px}
#form{display:flex;gap:8px;padding:12px;background:#16213e;border-top:1px solid #0f3460}
#input{flex:1;padding:10px 14px;border:1px solid #0f3460;border-radius:20px;background:#1a1a2e;color:#e0e0e0;font-size:15px;outline:none}
#input:focus{border-color:#e94560}
button{padding:10px 20px;background:#e94560;color:#fff;border:none;border-radius:20px;font-size:15px;cursor:pointer}
button:hover{background:#c73e54}
</style></head><body>
<div id="messages"></div>
<form id="form"><input id="input" placeholder="Say something..." autocomplete="off"><button type="submit">Send</button></form>
<script>
const div=document.getElementById('messages');
function render(msgs){
  div.innerHTML=msgs.map(m=>`<div class="msg ${m.role}"><div class="role">${m.role}</div>${m.content.replace(/</g,'&lt;')}</div>`).join('');
  div.scrollTop=div.scrollHeight;
}
async function refresh(){
  try{
    const r=await fetch('/chat');
    const msgs=await r.json();
    render(msgs);
  }catch(e){}
}
setInterval(refresh,2000);
refresh();
document.getElementById('form').onsubmit=async e=>{
  e.preventDefault();
  const inp=document.getElementById('input');
  const msg=inp.value.trim();
  if(!msg)return;
  inp.value='';
  const resp=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:msg})});
  if(resp.ok){
    div.insertAdjacentHTML('beforeend',`<div class="msg user"><div class="role">user</div>${msg.replace(/</g,'&lt;')}</div>`);
    div.scrollTop=div.scrollHeight;
    setTimeout(refresh,500);
  }
};
</script></body></html>"""

NUDGE_PREFIXES = [
    "Some time has passed",
    "Display is updated. Continue the rhythm",
    "You just woke up!",
    "Here is the latest photo",
    "The user pressed a button",
    "[Notification review]",
    "[Notification]",
    "The user approved your notification:",
]


class ChatHandler(BaseHTTPRequestHandler):
    orchestrator = None

    def log_message(self, format, *args):
        pass  # suppress default access logs

    def do_GET(self):
        if self.path == "/":
            self._serve_html()
        elif self.path == "/chat":
            self._get_messages()
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == "/chat":
            self._post_message()
        else:
            self.send_error(404)

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
            msgs = list(orch.ctx.get_messages())
        # Include queued messages that haven't been drained to context yet
        with orch.chat_queue_lock:
            for qm in orch.chat_queue:
                msgs.append({"role": "user", "content": qm})

        filtered = []
        for m in msgs:
            role = m.get("role")
            content = m.get("content", "")
            if role == "user":
                if isinstance(content, list):
                    continue  # image messages
                if not content or not content.strip():
                    continue
                if any(content.startswith(p) for p in NUDGE_PREFIXES):
                    continue
                filtered.append({"role": role, "content": content})
            elif role == "assistant":
                # Show only what was sent to the display
                for tc in m.get("tool_calls", []):
                    if tc.get("function", {}).get("name") == "update_display":
                        try:
                            args = json.loads(tc["function"]["arguments"])
                            display_text = args.get("text", "")
                            if display_text.strip():
                                filtered.append({"role": "assistant", "content": display_text})
                        except (json.JSONDecodeError, KeyError):
                            pass

        # Return last 50 messages
        filtered = filtered[-50:]
        data = json.dumps(filtered).encode()
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

        if orch.notification_store.has_pending_proposal():
            msg_lower = message.lower()
            if any(kw in msg_lower for kw in REJECTION_KEYWORDS):
                rejected = orch.notification_store.reject_pending()
                if rejected:
                    orch.last_fired_notification_id = None
                    print(f"[NOTIF] Proposal rejected via chat: {rejected['id']}")

        if orch.last_fired_notification_id:
            orch.notification_store.record_acknowledgment(orch.last_fired_notification_id)
            orch.last_fired_notification_id = None

        with orch.chat_queue_lock:
            orch.chat_queue.append(message)
        orch.chat_event.set()
        orch._reset_backoff()
        print(f"[CHAT] User message: {message[:100]}")

        data = json.dumps({"status": "ok"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    orch = Orchestrator()
    try:
        orch.run()
    finally:
        orch.cleanup()


if __name__ == "__main__":
    main()
