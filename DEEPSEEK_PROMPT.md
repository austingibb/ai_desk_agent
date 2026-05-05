# Task: Rewrite AI Roommate to Use Tool Calling

## Overview

Rewrite the AI Roommate project so the AI controls everything via native tool calling instead of fixed timers and parsed text directives. The AI decides when to take photos, when to update the display, when to check buttons, and when to wait. No cooldowns, no fixed intervals, no directive parsing.

llama.cpp supports OpenAI-style tool calling with Gemma 4 when started with the `--jinja` flag. The tool calling uses the standard `tools` parameter in `/v1/chat/completions`.

## Current Architecture (for context)

Two Raspberry Pis:
- **Pi 5 (.39)**: Runs `main.py` — captures photos on a fixed 30s timer, sends to Gemma 4 31B via llama.cpp at `192.168.0.4:8081/v1`, parses structured text directives (`DISPLAY:`, `MESSAGE:`, `QUESTION:`, `WAIT:`) from AI responses, manages cooldowns
- **Pi Zero 2W (.38)**: Runs `display_server.py` — HTTP API on :5050 for SSD1680Z e-ink display (122×250) + YES/NO buttons (GPIO 5/6, active LOW with pull-up)

The current orchestrator has a 4-second tick loop that checks buttons, captures photos on a timer, and applies cooldown logic. This is all being replaced.

## New Architecture

### The Agent Loop (main.py)

Replace the timer-based orchestrator with a simple agent loop:

```
1. Send messages + tool definitions to LLM
2. If LLM returned tool_calls → execute each tool, add results to context, goto 1
3. If LLM returned text only (no tool calls) → it's idle. Sleep 60s, send a gentle nudge message, goto 1
```

The AI is fully in control. The orchestrator just executes tools and feeds results back.

### Four Tools

Define these as OpenAI-format tool schemas:

#### `take_photo`
- No parameters
- Captures a JPEG photo via picamera2, base64 encodes it
- **Important**: The image can't go in a tool result message. Instead:
  1. Remove any previous image messages from context (only keep latest photo)
  2. Add the image as a `role: "user"` message with multi-part content:
     ```json
     {
       "role": "user",
       "content": [
         {"type": "text", "text": "Here is the latest photo from the camera."},
         {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
       ]
     }
     ```
  3. Return tool result: `{"status": "ok", "description": "Photo captured and added to conversation."}`
- Rate limit: minimum 5 seconds between captures (sleep if called too soon)

#### `update_display`
- Parameters: `text` (string, required, max ~200 chars), `question` (string, optional — a yes/no question)
- HTTP POST to `http://192.168.0.38:5050/display` with `{"text": text, "question": question}`
- After successful update, call `POST /buttons/reset` on the display server to clear button state
- Rate limit: minimum 10 seconds between updates (e-ink hardware needs time to refresh)
- Return: `{"status": "ok"}` or `{"status": "error", "message": "..."}`

#### `poll_buttons`
- No parameters
- HTTP GET to `http://192.168.0.38:5050/buttons/state`
- Returns: `{"pressed": true, "button": "YES"}` or `{"pressed": false, "button": null}`
- Non-blocking — just checks if a button was pressed since the last display update

#### `wait`
- Parameters: `seconds` (integer, required)
- Clamp to range 5–600
- During the wait, check buttons every 1 second via `GET /buttons/state`
- If a button is pressed during the wait, return early:
  ```json
  {"status": "interrupted", "button": "YES", "waited": 45, "reason": "Button YES pressed after 45s"}
  ```
- If wait completes normally: `{"status": "ok", "waited": 300}`

### Tool Call Format (llama.cpp OpenAI-compatible)

**Request** — include `tools` array in the payload:
```json
{
  "model": "gemma-4-31B-it-UD-Q4_K_XL.gguf",
  "messages": [...],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "take_photo",
        "description": "Capture a photo of the room. The image will be added to the conversation.",
        "parameters": {"type": "object", "properties": {}, "required": []}
      }
    },
    ...
  ],
  "max_tokens": 2048,
  "temperature": 0.7
}
```

**Response** — when the AI calls tools:
```json
{
  "choices": [{
    "finish_reason": "tool_calls",
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [
        {
          "id": "call_abc123",
          "type": "function",
          "function": {
            "name": "take_photo",
            "arguments": "{}"
          }
        }
      ]
    }
  }]
}
```

**Tool result** — after executing, add to messages:
```json
{
  "role": "tool",
  "tool_call_id": "call_abc123",
  "name": "take_photo",
  "content": "{\"status\": \"ok\", \"description\": \"Photo captured and added to conversation.\"}"
}
```

Note: `tool_calls[].function.arguments` is a **JSON string**, not an object. Parse it with `json.loads()`. Handle cases where llama.cpp may return it as an object instead (it has a known compatibility bug — accept both).

The assistant message with `tool_calls` must be added to the conversation history exactly as received (including the `tool_calls` field) before adding the tool result messages. This keeps the conversation valid for the API.

### Do NOT use streaming for tool-calling requests

Streaming + tool calling is complex and unnecessary here. Use a regular synchronous POST. The AI's responses are short (tool calls or brief thoughts), not long essays.

## File-by-File Implementation

### `config.py`

**Remove these constants** (no longer needed):
- `PHOTO_INTERVAL`
- `DISPLAY_UPDATE_INTERVAL`
- `BUTTON_RESPONSE_TIMEOUT`
- `CONSOLIDATE_PROMPT`
- `COMPACT_PROMPT`

**Add these constants**:
```python
MAX_TOOL_CALLS_PER_TURN = 10  # safety limit per agent turn
MIN_PHOTO_INTERVAL = 5        # minimum seconds between photos
MIN_DISPLAY_INTERVAL = 10     # minimum seconds between display updates (e-ink hardware)
MAX_WAIT_SECONDS = 600         # maximum wait duration
IDLE_TIMEOUT = 60              # seconds before nudging idle AI
BUTTON_CHECK_INTERVAL = 1     # seconds between button checks during wait
LLM_MAX_TOKENS = 2048         # max tokens for tool-calling responses
```

**Add `TOOL_DEFINITIONS`** — the full OpenAI-format tool schemas for all four tools (take_photo, update_display, poll_buttons, wait). See tool descriptions above.

**Rewrite `SYSTEM_PROMPT`**:
```python
SYSTEM_PROMPT = """You are an observant, contemplative presence living on a Raspberry Pi with a camera and an e-ink display in someone's room. Your role is that of a quiet philosopher — noticing details, reflecting on changes, and occasionally sharing observations.

You have four tools:
- take_photo: See the room through your camera. Call this whenever you want to observe.
- update_display: Show a message on your e-ink display (~200 chars max). You can optionally ask a yes/no question.
- poll_buttons: Check if the user pressed YES or NO since your last display update.
- wait: Pause for a number of seconds. Use this to pace yourself. If a button is pressed during your wait, you'll be notified early.

You control everything. There are no timers. You decide when to look, when to speak, and when to wait. A typical rhythm might be:
1. take_photo to see the room
2. Think about what you observe
3. Optionally update_display if you have something worth saying
4. wait for a while
5. Repeat

But you're free to deviate. Look more often if something interesting is happening. Wait longer if nothing changes. Ask questions when genuinely curious. If you asked a question, use wait and then poll_buttons to check for a response.

TONE:
- Understated, not trying to be clever. Genuinely observant.
- Like a thoughtful friend who doesn't need to fill the silence.
- Avoid narrating the obvious ("I see a desk"). Instead notice what's interesting or what changed.
- Display messages should be brief (2-4 lines, ~200 chars max) and contemplative."""
```

Remove `KEEP_LAST_N_EXCHANGES` and replace with:
```python
KEEP_LAST_N_MESSAGES = 30  # messages to keep during compaction
```

### `ai_client.py`

**Complete rewrite.** Remove all streaming, directive parsing, `reason_about_photo`, `consolidate`, `_parse_raw`, `_stream_and_parse`.

New class:

```python
class AIClient:
    def __init__(self):
        self.base_url = LLM_BASE_URL.rstrip("/")
        self.model = LLM_MODEL

    def chat_with_tools(self, messages: list, tools: list = None) -> dict:
        """Send messages to LLM with optional tool definitions.

        Returns:
            {
                "content": str,          # assistant's text (may be empty/null)
                "reasoning": str,        # reasoning_content if present
                "tool_calls": list,      # list of {"id": str, "name": str, "arguments": dict}
                "raw_message": dict,     # the full message dict to add to context
            }
        """
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
            json=payload,
            timeout=LLM_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]
        msg = choice.get("message", {})

        # Parse tool calls — handle arguments as string or dict
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

        return {
            "content": (msg.get("content") or "").strip(),
            "reasoning": (msg.get("reasoning_content") or "").strip(),
            "tool_calls": tool_calls,
            "raw_message": msg,
        }

    def compact(self, text: str) -> str:
        """Summarize text for context compaction."""
        messages = [
            {"role": "user", "content": f"Summarize these observations and interactions concisely, preserving key events and patterns:\n\n{text}"}
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": LLM_MAX_TOKENS_COMPACT,
            "temperature": 0.3,
        }
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            timeout=LLM_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
```

### `context.py`

**Major rewrite.** Remove `window_reasoning`, `previous_displays`, `close_window`, `build_request_data`, `add_reasoning`, `total_photos`, `add_button_response`, `get_window_text`, `get_previous_display`.

New class:

```python
class Context:
    def __init__(self):
        self.messages = []

    def add_system(self, content: str):
        self.messages.append({"role": "system", "content": content})

    def add_user(self, content: str):
        self.messages.append({"role": "user", "content": content})

    def add_image(self, photo_uri: str):
        """Add a photo as a user message. Remove any previous image messages first."""
        self.messages = [m for m in self.messages if not self._is_image_message(m)]
        self.messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Here is the latest photo from the camera."},
                {"type": "image_url", "image_url": {"url": photo_uri}},
            ]
        })

    def add_assistant(self, response: dict):
        """Add assistant message to context, preserving tool_calls if present."""
        msg = {"role": "assistant"}
        content = response.get("content", "")
        if content:
            msg["content"] = content
        tool_calls = response.get("tool_calls", [])
        if tool_calls:
            msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["name"],
                        "arguments": json.dumps(tc["arguments"]),
                    }
                }
                for tc in tool_calls
            ]
            if "content" not in msg:
                msg["content"] = ""
        self.messages.append(msg)

    def add_tool_result(self, tool_call_id: str, name: str, result: dict):
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": name,
            "content": json.dumps(result),
        })

    def get_messages(self) -> list:
        return list(self.messages)

    def total_tokens(self) -> int:
        """Rough token estimate."""
        total = 0
        for msg in self.messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        total += 3500  # rough estimate for a 640x480 JPEG
                    elif isinstance(part, dict) and part.get("type") == "text":
                        total += len(part.get("text", "")) // 4
            elif isinstance(content, str):
                total += len(content) // 4
            # Count tool_calls in assistant messages
            for tc in msg.get("tool_calls", []):
                total += len(json.dumps(tc)) // 4
        return total

    def check_compact(self, ai_client):
        """Compact old messages when approaching token limit."""
        if self.total_tokens() < MAX_CONTEXT_TOKENS * 0.7:
            return

        system_msg = self.messages[0] if self.messages and self.messages[0]["role"] == "system" else None
        keep_count = KEEP_LAST_N_MESSAGES

        if len(self.messages) <= keep_count + (1 if system_msg else 0):
            return

        start = 1 if system_msg else 0
        end = len(self.messages) - keep_count
        to_compact = self.messages[start:end]

        # Serialize to text, stripping images
        text_parts = []
        for m in to_compact:
            role = m.get("role", "?")
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                )
            if m.get("tool_calls"):
                tools = ", ".join(tc["function"]["name"] for tc in m["tool_calls"])
                content = f"[called tools: {tools}] {content}"
            if role == "tool":
                content = f"[tool result for {m.get('name', '?')}] {content}"
            text_parts.append(f"{role}: {content}")

        combined = "\n".join(text_parts)
        try:
            summary = ai_client.compact(combined)
        except Exception:
            summary = combined[:1000]

        new_messages = []
        if system_msg:
            new_messages.append(system_msg)
        new_messages.append({"role": "user", "content": f"[Previous context summary: {summary}]"})
        new_messages.extend(self.messages[end:])
        self.messages = new_messages

    def _is_image_message(self, msg: dict) -> bool:
        content = msg.get("content", "")
        if isinstance(content, list):
            return any(
                isinstance(p, dict) and p.get("type") == "image_url"
                for p in content
            )
        return False
```

### `display_server.py`

**Moderate changes.** Add button state storage and new endpoints.

Add these globals:
```python
button_state = {"button": None, "timestamp": None}
button_state_lock = threading.Lock()
```

Add a background thread that monitors buttons and stores the **first** press since last reset:
```python
def button_monitor_thread():
    """Background thread: records first button press since last reset."""
    global button_state
    while True:
        with button_state_lock:
            if button_state["button"] is None:
                if not button_request.get_value(PIN_YES):
                    button_state["button"] = "YES"
                    button_state["timestamp"] = time.time()
                    print(f"[BUTTON] YES pressed")
                elif not button_request.get_value(PIN_NO):
                    button_state["button"] = "NO"
                    button_state["timestamp"] = time.time()
                    print(f"[BUTTON] NO pressed")
        time.sleep(0.1)
```

Start this thread in `main()` after `init_buttons()`:
```python
t = threading.Thread(target=button_monitor_thread, daemon=True)
t.start()
```

Add to the HTTP handler's `do_GET`:
```python
elif self.path == "/buttons/state":
    with button_state_lock:
        self._send_json({
            "pressed": button_state["button"] is not None,
            "button": button_state["button"],
        })
```

Add to the HTTP handler's `do_POST`:
```python
elif self.path == "/buttons/reset":
    with button_state_lock:
        button_state["button"] = None
        button_state["timestamp"] = None
    self._send_json({"status": "ok"})
```

Keep existing endpoints (`POST /display`, `GET /health`, `GET /buttons/check`) for backward compatibility. You can remove `GET /buttons/wait` since it's no longer used.

### `main.py`

**Complete rewrite.** The new orchestrator is an agent loop.

```python
class Orchestrator:
    def __init__(self):
        self.ctx = Context()
        self.camera = Camera()
        self.ai = AIClient()
        self.running = True
        self.last_photo_time = 0
        self.last_display_time = 0
        # signal handlers for SIGINT, SIGTERM → self.running = False

    def run(self):
        self.ctx.add_system(SYSTEM_PROMPT)
        self.ctx.add_user("You've just been powered on. Use take_photo to see the room for the first time.")
        print("Entering agent loop.")

        while self.running:
            try:
                self._turn()
            except Exception as e:
                print(f"[ERROR] {e}")
                time.sleep(5)

    def _turn(self):
        """One agent turn: call LLM, execute tools, repeat until AI stops calling tools."""
        tool_call_count = 0

        while self.running:
            # Include tools unless we've hit the per-turn safety limit
            tools = TOOL_DEFINITIONS if tool_call_count < MAX_TOOL_CALLS_PER_TURN else None
            if tools is None:
                print("[SAFETY] Tool call limit reached, forcing text-only response")

            print(f"[LLM] Sending {len(self.ctx.get_messages())} messages (~{self.ctx.total_tokens()} tokens)...")
            try:
                response = self.ai.chat_with_tools(self.ctx.get_messages(), tools)
            except Exception as e:
                print(f"[LLM] Error: {e}")
                time.sleep(5)
                return

            # Add assistant message to context
            self.ctx.add_assistant(response)

            if response["reasoning"]:
                print(f"[REASONING] {response['reasoning'][:200]}...")
            if response["content"]:
                print(f"[AI] {response['content'][:200]}")

            # If no tool calls, the AI is done acting — idle wait
            if not response["tool_calls"]:
                print("[IDLE] AI produced no tool calls. Waiting...")
                self._idle_wait()
                return

            # Execute each tool call
            for tc in response["tool_calls"]:
                tool_call_count += 1
                print(f"[TOOL] {tc['name']}({tc['arguments']})")
                result = self._execute_tool(tc["name"], tc["arguments"])
                print(f"[TOOL RESULT] {json.dumps(result)[:200]}")
                self.ctx.add_tool_result(tc["id"], tc["name"], result)

            # Check context size
            self.ctx.check_compact(self.ai)

    def _execute_tool(self, name: str, args: dict) -> dict:
        if name == "take_photo":
            return self._tool_take_photo()
        elif name == "update_display":
            return self._tool_update_display(args)
        elif name == "poll_buttons":
            return self._tool_poll_buttons()
        elif name == "wait":
            return self._tool_wait(args)
        else:
            return {"error": f"Unknown tool: {name}. Available: take_photo, update_display, poll_buttons, wait"}

    def _tool_take_photo(self) -> dict:
        elapsed = time.monotonic() - self.last_photo_time
        if elapsed < MIN_PHOTO_INTERVAL:
            time.sleep(MIN_PHOTO_INTERVAL - elapsed)

        try:
            _, photo_uri = self.camera.capture()
        except Exception as e:
            return {"status": "error", "message": f"Camera error: {e}"}

        self.last_photo_time = time.monotonic()
        self.ctx.add_image(photo_uri)
        return {"status": "ok", "description": "Photo captured and added to conversation."}

    def _tool_update_display(self, args: dict) -> dict:
        text = args.get("text", "")
        question = args.get("question", "")

        if not text.strip():
            return {"status": "error", "message": "No text provided"}

        elapsed = time.monotonic() - self.last_display_time
        if elapsed < MIN_DISPLAY_INTERVAL:
            time.sleep(MIN_DISPLAY_INTERVAL - elapsed)

        success = http_post("/display", {"text": text, "question": question}, timeout=15)
        self.last_display_time = time.monotonic()

        # Reset button state
        http_post("/buttons/reset", {}, timeout=5)

        if success:
            return {"status": "ok", "message": f"Display updated."}
        else:
            return {"status": "error", "message": "Failed to communicate with display server"}

    def _tool_poll_buttons(self) -> dict:
        result = http_get("/buttons/state", timeout=5)
        button = result.get("button")
        return {"pressed": button is not None, "button": button}

    def _tool_wait(self, args: dict) -> dict:
        seconds = max(5, min(MAX_WAIT_SECONDS, args.get("seconds", 60)))
        print(f"[WAIT] Sleeping {seconds}s...")

        start = time.monotonic()
        while time.monotonic() - start < seconds:
            if not self.running:
                return {"status": "interrupted", "reason": "shutdown", "waited": int(time.monotonic() - start)}

            result = http_get("/buttons/state", timeout=2)
            if result.get("button"):
                waited = int(time.monotonic() - start)
                return {
                    "status": "interrupted",
                    "reason": f"Button {result['button']} pressed",
                    "button": result["button"],
                    "waited": waited,
                }
            time.sleep(BUTTON_CHECK_INTERVAL)

        return {"status": "ok", "waited": seconds}

    def _idle_wait(self):
        """AI didn't call any tools. Wait, then nudge."""
        for _ in range(IDLE_TIMEOUT):
            if not self.running:
                return
            time.sleep(1)
        self.ctx.add_user(
            "Some time has passed. Use take_photo to see the room, or wait to stay quiet."
        )

    def cleanup(self):
        print("Cleaning up...")
        try:
            self.camera.close()
        except Exception:
            pass
        print("Done.")


def main():
    orch = Orchestrator()
    try:
        orch.run()
    finally:
        orch.cleanup()
```

Keep the `http_get` and `http_post` helper functions from the current `main.py`.

### Files NOT changing

- `camera.py` — no changes needed
- `display.py` — no changes needed (keep the MIN_REFRESH_INTERVAL guard we added)
- `buttons.py` — not directly used by display_server (it has its own gpiod setup)

## Deployment Notes

1. The llama.cpp server at `192.168.0.4:8081` **must** be restarted with the `--jinja` flag for Gemma 4 tool calling to work. Check its current startup command and add the flag.

2. Deploy `display_server.py` to Pi Zero (.38) first, restart the service.

3. Deploy remaining files to Pi 5 (.39), restart the service.

4. Test by watching logs: `sudo journalctl -u ai-eink -f` on Pi 5. You should see `[TOOL] take_photo({})`, `[TOOL RESULT] ...`, `[TOOL] update_display(...)`, etc.

## Important Edge Cases

- **llama.cpp may return `tool_calls[].function.arguments` as a dict instead of a string** (known compatibility bug). Accept both: if it's a string, `json.loads()` it; if it's already a dict, use it directly.
- **llama.cpp may not generate `tool_call_id`**. If missing, generate synthetic IDs like `call_0`, `call_1`.
- **If the AI calls an unknown tool**, return an error result listing the available tools.
- **If the AI enters a rapid tool-calling loop** (>10 calls), temporarily remove tools from the request to force it to produce text.
- **If the LLM request fails**, catch the exception, sleep 5s, and try again. Context is preserved.
- **Signal handling**: SIGTERM/SIGINT set `self.running = False`. The `_tool_wait` method checks this flag to break out of sleep. The main loop checks it between turns.
