# AI E-Ink Friend

An autonomous AI friend running on two Raspberry Pis — it watches the room, shares thoughts on an e-ink display, and chats with you. Uses DeepSeek on OpenRouter as the brain and local Gemma 4 for vision.

## Hardware

| Device | Role |
|--------|------|
| **Pi 5** | Orchestrator — runs the AI agent loop, camera, web chat server |
| **Pi Zero 2W** | Display server — drives SSD1680Z e-ink (122×250) + two GPIO buttons |
| **Any machine with a GPU** | Vision LLM via llama.cpp — runs Gemma 4 31B for photo descriptions. Anything faster than a Pi 5 works here. |

## How it works

The AI runs in an autonomous agent loop with a **two-model architecture**:

- **DeepSeek on OpenRouter** — the brain. Handles reasoning, tool calling, conversation, and notification management. Text-only, no images.
- **Local Gemma 4 31B** — vision-only. A background thread captures photos every ~3 minutes, sends them to Gemma for a text description, and caches the result.

When the brain calls `take_photo`, it gets the cached description instantly — no round-trip to the vision model during the agent loop. For moments when it genuinely needs to see what's happening *right now* (e.g., "did the user actually do what they said?"), `capture_photo` takes a new photo and blocks until the vision model responds (up to 120s).

### Agent loop

1. The orchestrator sends conversation history + tool definitions to DeepSeek
2. DeepSeek decides what to do: check the room, update the display, wait, search the web, or just idle
3. Tools are executed and results fed back into the conversation
4. Repeat — no timers, no hardcoded logic, the AI is in full control

### Tools

| Tool | What it does |
|------|-------------|
| `take_photo` | Returns a cached text description of the room (photos taken automatically every ~3 min). Instant. |
| `capture_photo` | Takes a new photo and waits for the vision model to describe it. Slow (up to 120s) — use sparingly. |
| `update_display` | Renders a SHORT message (~140 chars max) on the e-ink display with a timestamp |
| `send_chat_message` | Sends a LONG message to the chat UI (no length limit). The e-ink shows a short preview. |
| `wait` | Pauses — polls for button presses, chat messages, notifications, and review intervals |
| `propose_notification` | Proposes a recurring notification for user approval via button press |
| `schedule_notification` | Schedules when a notification fires next (interval or deferral) |
| `delete_notification` | Permanently deletes a notification by ID |
| `update_vision_requests` | Changes what the vision model looks for when describing the scene |
| Brave Search tools | Web, local, image, video, news search + summarizer via MCP |

### Interaction

- **Web chat** — Password-protected login with session cookie (7 day expiry). Type messages to the AI, see its display responses with timestamps. Append-only rendering — scroll up to read history without being dragged to bottom.
- **Physical buttons** (GPIO 5/6) — Press either button to nudge the AI to say something new, or approve a proposed notification
- **Brave Search** — MCP integration for web search, news, images, and more

### Chat security

- Password login via `CHAT_PASSWORD` env var (default: `admin`)
- Random 32-byte session token, stored in an `HttpOnly` cookie
- Session lasts 7 days before re-login required
- Optional HTTPS via [mkcert](https://github.com/FiloSottile/mkcert) — set `CHAT_USE_HTTPS=1` and provide `SSL_CERT_FILE` / `SSL_KEY_FILE`
- Cookie gets `Secure` flag when HTTPS is enabled

### Wait

The AI controls wait duration directly (clamped 10s–30 min). Waits poll for button presses and chat messages every second — any interaction interrupts the wait early.

### Context management

- Conversation persisted to `context.json` across restarts
- System prompt refreshed from code on every restart
- Token counting at ~4 chars/token
- Auto-compaction at 150 messages — summarizes old messages (via DeepSeek or local Gemma) into a summary, keeps last 30
- OpenRouter message pairing enforced via `_repair_pairing()` — fixes orphan tool results, sandwiched messages, and unmatched tool calls

### Notifications

The AI can propose recurring notifications (stretch reminders, "it's getting late", etc.):
- Proposed via `propose_notification` tool during periodic reviews
- User approves via button press, rejects via chat ("no", "stop", etc.)
- Category scoring tracks what the user likes/dislikes
- Decay system expires unacknowledged notifications over time

### Sound effects

Non-blocking PulseAudio sounds play on key events: thinking, taking a photo, updating the display, searching, waiting.

## Configuration

Set via environment variables or `.env` file:

### Brain LLM (DeepSeek/OpenRouter)

| Variable | Default | Notes |
|----------|---------|-------|
| `LLM_BASE_URL` | `https://openrouter.ai/api/v1` | Any OpenAI-compatible API |
| `LLM_MODEL` | `deepseek/deepseek-chat` | Model name |
| `LLM_API_KEY` | _(empty)_ | Required for OpenRouter |

### Vision LLM (local Gemma/llama.cpp)

| Variable | Default | Notes |
|----------|---------|-------|
| `VISION_BASE_URL` | `http://<llama-server>:8081/v1` | Local llama.cpp server |
| `VISION_MODEL` | `gemma-4-31B-it-UD-Q4_K_XL.gguf` | Vision model name |
| `VISION_API_KEY` | _(empty)_ | Optional |
| `VISION_POLL_INTERVAL` | `180` | Seconds between photo captures |

### Chat & Security

| Variable | Default | Notes |
|----------|---------|-------|
| `CHAT_PASSWORD` | `admin` | Password for web chat login |
| `CHAT_USE_HTTPS` | `0` | Set to `1` to enable HTTPS |
| `SSL_CERT_FILE` | `cert.pem` (project dir) | Path to TLS certificate |
| `SSL_KEY_FILE` | `key.pem` (project dir) | Path to TLS private key |

### Other

| Variable | Default | Notes |
|----------|---------|-------|
| `ENABLE_CAMERA` | `1` | Set to `0` to disable camera/vision tools |
| `COMPACT_AFTER_N_MESSAGES` | `150` | Trigger compaction threshold |
| `REVIEW_INTERVAL` | `1800` | Seconds between notification reviews |

## HTTPS Setup (optional)

```bash
# On your dev machine (one time)
brew install mkcert
sudo mkcert -install
mkcert <pi5-ip> localhost

# Copy certs to Pi
rsync -avz <pi5-ip>+1.pem <pi5-ip>+1-key.pem user@<pi5-ip>:~/.config/certs/

# Set in .env on Pi
CHAT_USE_HTTPS=1
SSL_CERT_FILE=/home/user/.config/certs/<pi5-ip>+1.pem
SSL_KEY_FILE=/home/user/.config/certs/<pi5-ip>+1-key.pem
```

## Deployment

### Pi 5 (orchestrator)

```bash
cd ~/ai_eink
pip install -r requirements.txt
sudo systemctl enable --now ai-eink
```

### Pi Zero 2W (display)

```bash
cd ~/ai_eink
pip install -r requirements-display.txt
sudo systemctl enable --now display-server
```

### Quick deploy (from dev machine)

```bash
ssh user@<pi5-ip> 'cd ~/ai_eink && git pull && sudo systemctl restart ai-eink'
ssh user@<pizero-ip> 'cd ~/ai_eink && git pull && sudo systemctl restart display-server'
```

### Watching logs

```bash
ssh user@<pi5-ip> 'sudo journalctl -u ai-eink -f'
```

## Files

| File | Runs on | Purpose |
|------|---------|---------|
| `main.py` | Pi 5 | Agent loop, tool execution, vision thread, web chat server |
| `config.py` | Both | All constants, system prompt, tool definitions |
| `context.py` | Pi 5 | Conversation history, token counting, compaction, pairing repair |
| `ai_client.py` | Pi 5 | `AIClient` (DeepSeek) + `VisionClient` (local Gemma) |
| `camera.py` | Pi 5 | Picamera2 capture at 2304×1296 → 640px downscale → base64 JPEG |
| `mcp_client.py` | Pi 5 | Brave Search MCP integration (JSON-RPC/SSE) |
| `notifications.py` | Pi 5 | Notification proposals, scoring, decay, review summaries |
| `sounds.py` | Pi 5 | Non-blocking PulseAudio sound playback |
| `display_server.py` | Pi Zero 2W | HTTP API on :5050 for display + button monitoring |
| `display.py` | Pi Zero 2W | SSD1680Z e-ink driver, PIL text rendering |
| `buttons.py` | Pi Zero 2W | GPIO button reading (gpiod v2) |
| `requests_for_image_model.md` | Pi 5 | Dynamic instructions for vision model (editable by AI) |
