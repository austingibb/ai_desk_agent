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
    SYSTEM_PROMPT,
    TOOL_DEFINITIONS,
    MAX_TOOL_CALLS_PER_TURN,
    MIN_PHOTO_INTERVAL,
    MIN_DISPLAY_INTERVAL,
    MAX_WAIT_SECONDS,
    IDLE_TIMEOUT,
    BUTTON_CHECK_INTERVAL,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
    CHAT_SERVER_PORT,
    BACKOFF_BASE,
    BACKOFF_MAX,
)
from context import Context
from camera import Camera
from ai_client import AIClient
from mcp_client import MCPClient
from sounds import play as play_sound

LOG_FILE = "/home/austingibb/ai_eink/verbose.log"


def log(msg: str):
    print(msg)
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
        self.running = True
        self.last_photo_time = 0
        self.last_display_time = 0
        self.chat_event = threading.Event()
        self.ctx_lock = threading.Lock()
        self.backoff = BACKOFF_BASE
        self.mcp_tools = []
        self.mcp = None

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
        print("Init AI client...")
        self._start_chat_server()
        with self.ctx_lock:
            if self.ctx.load():
                print("Resuming from saved context.")
                self.ctx.add_user("You just woke back up after a restart! Use take_photo to see the room and pick up where you left off.")
            else:
                self.ctx.add_system(SYSTEM_PROMPT)
                self.ctx.add_user("You just woke up! Use take_photo to see the room and say hi.")
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
            tools = TOOL_DEFINITIONS
            if self.mcp_tools:
                tools = TOOL_DEFINITIONS.copy()
                tools.extend(self.mcp_tools)

            with self.ctx_lock:
                messages = self.ctx.get_messages()
                tokens = self.ctx.total_tokens()
            print(f"[LLM] Sending {len(messages)} messages (~{tokens} tokens)...")
            play_sound("thinking")
            try:
                response = self.ai.chat_with_tools(messages, tools)
            except Exception as e:
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

            for tc in response["tool_calls"]:
                tool_call_count += 1
                last_tool_name = tc["name"]
                print(f"[TOOL] {tc['name']}({tc['arguments']})")
                result = self._execute_tool(tc["name"], tc["arguments"])
                print(f"[TOOL RESULT] {json.dumps(result)[:200]}")
                log(f"[TOOL RESULT] {json.dumps(result)}")
                with self.ctx_lock:
                    self.ctx.add_tool_result(tc["id"], tc["name"], result)

            with self.ctx_lock:
                self.ctx.check_compact(self.ai)

    def _execute_tool(self, name: str, args: dict) -> dict:
        if name == "take_photo":
            play_sound("take_photo")
            return self._tool_take_photo()
        elif name == "update_display":
            play_sound("update_display")
            return self._tool_update_display(args)
        elif name == "wait":
            play_sound("wait")
            return self._tool_wait(args)
        else:
            if self.mcp:
                play_sound("search")
                try:
                    return self.mcp.call_tool(name, args)
                except Exception as e:
                    return {"error": f"MCP tool '{name}' failed: {e}"}
            return {"error": f"Unknown tool: {name}. Available: take_photo, update_display, wait"}

    def _tool_take_photo(self) -> dict:
        elapsed = time.monotonic() - self.last_photo_time
        if elapsed < MIN_PHOTO_INTERVAL:
            time.sleep(MIN_PHOTO_INTERVAL - elapsed)

        try:
            _, photo_uri = self.camera.capture()
        except Exception as e:
            return {"status": "error", "message": f"Camera error: {e}"}

        self.last_photo_time = time.monotonic()
        with self.ctx_lock:
            self.ctx.add_image(photo_uri)
        return {"status": "ok", "description": "Photo captured and added to conversation."}

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

            result = http_get("/buttons/state", timeout=2)
            if result.get("button"):
                self._reset_backoff()
                http_post("/buttons/reset", {}, timeout=5)
                waited = int(time.monotonic() - start)
                with self.ctx_lock:
                    self.ctx.add_user("The user pressed a button — they want you to say something!")
                print(f"[WAIT] Interrupted by button press after {waited}s")
                return {
                    "status": "interrupted",
                    "reason": f"Button {result['button']} pressed — injected nudge",
                    "button": result["button"],
                    "waited": waited,
                }
            time.sleep(BUTTON_CHECK_INTERVAL)

        self.backoff = min(self.backoff * 2, BACKOFF_MAX)
        print(f"[WAIT] Completed. Next backoff: {self.backoff}s")
        return {"status": "ok", "waited": seconds}

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
                with self.ctx_lock:
                    self.ctx.add_user("The user pressed a button — they want you to say something!")
                print("[IDLE] Interrupted by button press")
                return
            time.sleep(1)
        with self.ctx_lock:
            self.ctx.add_user(
                "Some time has passed. Use take_photo to see the room, or wait to stay quiet."
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
        with orch.ctx_lock:
            orch.ctx.add_user(message)
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
