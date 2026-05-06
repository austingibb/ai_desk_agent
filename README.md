# AI E-Ink Friend

An autonomous AI friend running on two Raspberry Pis — it watches the room, shares thoughts on an e-ink display, and chats with you.

## Hardware

| Device | Role |
|--------|------|
| **Pi 5 (CM5, .39)** | Orchestrator — runs the AI agent loop, camera, web chat server |
| **Pi Zero 2W (.38)** | Display server — drives SSD1680Z e-ink (122×250) + two GPIO buttons |
| **Separate machine (.4:8081)** | LLM via llama.cpp (OpenAI-compatible API) — defaults to Gemma 4 31B |

## How it works

The AI runs in an autonomous agent loop using OpenAI-style **tool calling**:

1. The orchestrator sends conversation history + tool definitions to the LLM
2. The LLM decides what to do: take a photo, update the display, or wait
3. Tools are executed and results fed back into the conversation
4. Repeat — no timers, no hardcoded logic, the AI is in full control

### Tools

| Tool | What it does |
|------|-------------|
| `take_photo` | Captures a 640×480 photo via Picamera2, adds it to the conversation |
| `update_display` | Renders text (~140 chars max) on the e-ink display with a timestamp |
| `wait` | Pauses — polls for button presses or chat messages every second |

### Interaction

- **Web chat** (`http://192.168.0.39:8080`) — Type messages to the AI, see its display responses
- **Physical buttons** (GPIO 5/6) — Press either button to nudge the AI to say something new
- **Brave Search** — Optional MCP integration for web search, news, images

### Wait backoff

Waits use exponential backoff to avoid spamming the display:

- Starts at 10s → doubles on each peaceful completion (20s, 40s, 80s, 160s, capped at 180s)
- Any user interaction (chat or button) resets it back to 10s

### Context management

- Conversation persisted to `context.json` across restarts
- Rough token counting (chars ÷ 4, images ≈ 3500 tokens)
- Auto-compaction at 70% of the 64k token limit — compresses old messages into a summary
- Only the latest photo is kept in context

### Sound effects

Non-blocking PulseAudio sounds play on key events: thinking, taking a photo, updating the display, searching, waiting.

## LLM configuration

Set via environment variables or `.env` file:

| Variable | Default | Notes |
|----------|---------|-------|
| `LLM_BASE_URL` | `http://192.168.0.4:8081/v1` | Any OpenAI-compatible API |
| `LLM_MODEL` | `gemma-4-31B-it-UD-Q4_K_XL.gguf` | Model name |
| `LLM_API_KEY` | _(empty)_ | Required for OpenRouter, optional for local |

Gemma-specific quirks (control token stripping, inline `<|tool_call|>` parsing) are applied automatically when the model name contains "gemma". Non-Gemma models get standard OpenAI tool call passthrough.

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

### Watching logs

```bash
ssh austingibb@192.168.0.39 'sudo journalctl -u ai-eink -f'
```

## Files

| File | Runs on | Purpose |
|------|---------|---------|
| `main.py` | Pi 5 | Agent loop, tool execution, web chat server |
| `config.py` | Both | All constants, system prompt, tool definitions |
| `context.py` | Pi 5 | Conversation history, token counting, compaction |
| `ai_client.py` | Pi 5 | LLM HTTP client with tool-call parsing |
| `camera.py` | Pi 5 | Picamera2 capture → base64 JPEG |
| `mcp_client.py` | Pi 5 | Brave Search MCP integration (JSON-RPC/SSE) |
| `sounds.py` | Pi 5 | Non-blocking PulseAudio sound playback |
| `display_server.py` | Pi Zero 2W | HTTP API on :5050 for display + button monitoring |
| `display.py` | Pi Zero 2W | SSD1680Z e-ink driver, PIL text rendering |
| `buttons.py` | Pi Zero 2W | Direct GPIO button reading (gpiod) |
