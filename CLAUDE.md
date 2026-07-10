# AI E-Ink Friend

Two Raspberry Pis running an autonomous AI agent that observes the room through a camera, chats with the user, and displays messages on an e-ink screen. Uses a two-model architecture: DeepSeek on OpenRouter for reasoning/tool calling, and local Qwen3.6 27B on llama.cpp for vision and compaction.

## Architecture

```
Pi 5 — Orchestrator (main.py)
├── camera.py        → Picamera2 capture (2304×1296 full FOV → 640px downscale)
├── ai_client.py     → AIClient (DeepSeek/OpenRouter) + VisionClient (local Qwen/llama.cpp)
├── context.py       → Message history, timestamps, token counting, compaction, pairing repair
├── mcp_client.py    → Brave Search MCP integration (JSON-RPC over SSE/HTTP)
├── sounds.py        → Non-blocking PulseAudio playback for tool events
├── notifications.py → Notification proposals, approval/rejection, decay scoring
├── caffeine.py      → Append-only drink log (drinks.json, 24h retention)
├── presence.py      → ActiveTracker: at-desk boolean from motion/chat/button activity
├── status_publisher.py → Publishes {active, drinks} JSON to public S3 for aarg.dev
└── chat server :8080  → Web UI for user to type messages

Pi Zero 2W — Display Server (display_server.py :5050)
├── display.py       → SSD1680Z e-ink driver (122×250 via SPI)
└── buttons.py       → GPIO button polling (YES=5, NO=6, active LOW)

LLM Server (llama.cpp :8080)
└── llama.cpp running Qwen3.6 27B Q4 — vision descriptions + compaction fallback
```

## Two-Model Architecture

- **DeepSeek (OpenRouter)** — the brain. Reasoning, tool calling, display decisions, notification management. Text-only.
- **Local Qwen3.6 27B (llama.cpp)** — vision-only. Background thread captures photos every 3 min, sends to Qwen for description, caches result.

DeepSeek accesses the scene via a `take_photo` tool that returns the latest cached text description (instant, no round-trip to local LLM at call time). For moments when the AI genuinely needs to see what's happening right now, `capture_photo` takes a new photo and blocks until the vision model responds (up to 120s). Compaction uses DeepSeek if API key is set, otherwise falls back to local Qwen.

## Agent Loop (main.py Orchestrator._turn)

1. Drain chat queue into context (thread-safe)
2. Run `_repair_pairing()` to fix OpenRouter message format violations
3. Build tools list (core + MCP tools)
4. Estimate tokens, trigger compaction if needed
5. Send messages + tool definitions to DeepSeek
6. Store assistant response in context
7. Execute each tool call, store results
8. After `update_display`, enforce a wait (unless DeepSeek already called `wait`)
9. Check compaction after all tool results
10. If no tool calls → idle timeout → nudge → restart

## Background Vision Loop

A daemon thread (`_start_vision_loop`) runs independently:
1. Every `VISION_POLL_INTERVAL` (180s): capture photo via Picamera2
2. Send base64 JPEG to local Qwen via `VisionClient.describe()`
3. Cache result in `self.latest_scene` (protected by `scene_lock`)
4. Save debug JPEG to `debug_images/` (24h rolling window)
5. Retry up to 3 times on empty response (the vision model intermittently returns empty)

When DeepSeek calls `take_photo`, it gets the cached description instantly.

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | Orchestrator loop, tool execution, vision thread, chat server, signal handling |
| `config.py` | All constants, system prompt, tool definitions, `ENABLE_CAMERA` flag |
| `context.py` | Message store with timestamps, compaction, `_repair_pairing()` for OpenRouter |
| `ai_client.py` | `AIClient` (DeepSeek/OpenRouter) + `VisionClient` (local Qwen/llama.cpp) |
| `camera.py` | Picamera2 capture at 2304×1296, downscale to 640px, JPEG encode |
| `mcp_client.py` | Brave Search MCP client (JSON-RPC over SSE/HTTP) |
| `notifications.py` | Notification proposals, approval/rejection, decay scoring, review summaries |
| `caffeine.py` | `DrinkStore` — append-only caffeine log in `drinks.json`, pruned to 24h |
| `presence.py` | `ActiveTracker` — "at desk" boolean (activity within 5 min, debounced) |
| `status_publisher.py` | Daemon thread uploading `{active, drinks}` to S3 (public feed for aarg.dev) |
| `setup-aws.sh` | One-time bootstrap: bucket, public-read policy, CORS, scoped IAM user |
| `sounds.py/sounds/` | PulseAudio sound effects for tool events |
| `display_server.py` | HTTP API for display updates, button state, health checks |
| `display.py` | E-ink hardware driver (PIL text rendering) |
| `buttons.py` | GPIO button reading via gpiod v2 |
| `requests_for_image_model.md` | Dynamic instructions for what the vision model looks for |

## Tools

**Core tools** (defined in `config.py` TOOL_DEFINITIONS):
- `take_photo` — returns cached text description from background vision thread (instant)
- `capture_photo` — takes a new photo and blocks until the vision model describes it (up to 120s). Use sparingly — only for moments you genuinely need fresh info.
- `update_display` — show message on e-ink (~140 chars max)
- `send_chat_message` — send longer message to the chat UI (no length limit). E-ink shows a short preview.
- `wait` — pause with button/chat interruption polling
- `propose_notification`, `schedule_notification`, `delete_notification` — manage recurring notifications
- `log_drink` — append a caffeine drink (mg, label, optional minutes_ago) to the public feed; agent converts drink names → mg via the reference table in the system prompt
- `update_vision_requests` — modify what the vision model looks for

**MCP tools** (Brave Search, via `mcp_client.py`):
- `brave_web_search`, `brave_local_search`, `brave_image_search`, `brave_video_search`, `brave_news_search`, `brave_summarizer`

When `ENABLE_CAMERA=0`, `take_photo`, `capture_photo`, and `update_vision_requests` are excluded from tools and system prompt.

## Configuration

All in `config.py`. Key constants:

### Brain LLM (DeepSeek/OpenRouter)
- `LLM_BASE_URL` (default: `https://openrouter.ai/api/v1`)
- `LLM_API_KEY` (env var, required for OpenRouter)
- `LLM_MODEL` (default: `deepseek/deepseek-chat`)
- `LLM_MAX_TOKENS` (2048), `LLM_MAX_TOKENS_COMPACT` (1024), `LLM_TIMEOUT` (120s)

### Vision LLM (local Qwen/llama.cpp)
- `VISION_BASE_URL` (default: `http://<llama-server>:8080/v1`)
- `VISION_MODEL` (default: `Qwen3.6-27B-UD-Q4_K_XL.gguf`)
- `VISION_POLL_INTERVAL` (180s), `VISION_TIMEOUT` (60s)

### Context & Compaction
- `COMPACT_AFTER_N_MESSAGES` (150, env override) — trigger compaction by message count
- `KEEP_LAST_N_MESSAGES` (30) — messages kept after compaction
- `MAX_CONTEXT_TOKENS` (64000), `TOKEN_ESTIMATE_DIVISOR` (4 chars/token)

### Agent Behavior
- `BACKOFF_BASE` (10s), `BACKOFF_MAX` (900s) — wait backoff (triples each cycle, resets on interaction)
- `MAX_TOOL_CALLS_PER_TURN` (10), `MIN_DISPLAY_INTERVAL` (10s), `IDLE_TIMEOUT` (60s)

### Notifications
- `REVIEW_INTERVAL` (1800s / 30 min), `MAX_PROPOSAL_INTERVAL` (7200s / 2h between proposals)
- `MAX_FIRINGS_PER_HOUR` (1), `CATEGORY_COOLDOWN_REVIEWS` (3)

### Chat
- `CHAT_PASSWORD` (env, default `admin`) — password for web UI login
- `CHAT_SESSION_DAYS` (7) — session cookie expiry in days
- `CHAT_USE_HTTPS` (env, default 0) — enable TLS via mkcert certs
- `SSL_CERT_FILE`, `SSL_KEY_FILE` — paths to TLS certificate and key

### Caffeine Status Feed (public!)
- `ENABLE_STATUS_PUBLISH` (env, default 1), `STATUS_S3_BUCKET` (env, empty = disabled)
- `STATUS_S3_KEY` (`caffeine.json`), `STATUS_PUBLISH_INTERVAL` (45s heartbeat)
- `ACTIVE_WINDOW_SECONDS` (300) — no motion/chat/button for 5 min → `active: false`
- AWS creds via `.env` on Pi 5 only (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`); boto3 reads them from the environment. One-time infra via `setup-aws.sh`.
- Feed shape (aarg.dev depends on it): `{"active": bool, "drinks": [{"t": epoch_ms, "mg": int}]}` — raw events, no decay math, 24h retention, never future timestamps.

### Hardware
- `ENABLE_CAMERA` (env, default 1) — toggle camera/vision features
- `CAMERA_WIDTH` (2304), `CAMERA_HEIGHT` (1296) — full sensor FOV
- `DISPLAY_WIDTH` (250), `DISPLAY_HEIGHT` (122) — SSD1680Z e-ink

## Context & Compaction

Messages stored in OpenAI format with `_ts` (timestamp) field. `get_messages()` injects human-readable timestamps like `[Wed 14:30:22]` into content before sending to the LLM.

Compaction triggers at `COMPACT_AFTER_N_MESSAGES` (150). Summarizes everything except system prompt and last 30 messages into a `[Previous context summary: ...]` message. Uses `_find_safe_end()` to avoid splitting assistant/tool pairs. Compaction done by DeepSeek (with local Qwen fallback).

## OpenRouter Message Pairing

OpenRouter requires strict format: assistant messages with `tool_calls` must be immediately followed by matching `tool` result messages. `_repair_pairing()` in context.py fixes three violation types:
1. Orphan tool messages (no matching assistant)
2. Sandwiched non-tool messages between assistant and its tool results
3. Unfulfilled tool_calls (trims from assistant's tool_calls list)

Chat messages are queued (`chat_queue`) and drained at safe points to avoid breaking pairing.

## Chat Server

Web UI on `:8080`. Password-protected login with session cookie. Supports optional HTTPS via mkcert certificates.

- **Auth**: Password from `CHAT_PASSWORD` env var (default `admin`). Random 32-byte session token in `HttpOnly` cookie, expires after `CHAT_SESSION_DAYS` (7 days). Login page at `/login` with password form, redirects to `/` on success.
- **HTTPS**: Set `CHAT_USE_HTTPS=1` and provide `SSL_CERT_FILE`/`SSL_KEY_FILE` (mkcert certs). Cookie gains `Secure` flag. Access via `https://<pi5-ip>:8080`.
- **GET `/`** → chat HTML (requires auth, else login page)
- **GET `/chat`** → last 50 filtered messages as JSON (with timestamps as metadata). Deduplicates user messages between context and chat queue. Requires auth (else 401).
- **POST `/chat`** → queues message and signals the agent loop. Detects notification rejection keywords ("no", "stop", "cancel").
- **POST `/login`** → validates password, sets session cookie, redirects to `/`
- **Rendering**: Client appends new messages only (no full DOM replacement). Tracks rendered messages via `Set` of content signatures. Auto-scrolls only when user is at bottom — scroll up to read history without interruption.

## Add New Tool

1. Add tool definition to `TOOL_DEFINITIONS` in `config.py`
2. Add handler in `main.py._execute_tool()`
3. If camera-related: add name to `CAMERA_TOOL_NAMES` in `config.py`
4. If external tool: add to `mcp_client.py`

## Common Tasks

**Deploy**: Commit + push, then SSH to each Pi and pull + restart services:
```bash
ssh user@<pi5-ip> 'cd ~/ai_desk_agent && git pull && sudo systemctl restart ai-eink'
ssh user@<pizero-ip> 'cd ~/ai_desk_agent && git pull && sudo systemctl restart display-server'
```

**Watch logs**: `ssh user@<pi5-ip> 'sudo journalctl -u ai-eink -f'`

**Test locally**: `python3 -c "import py_compile; py_compile.compile('main.py', doraise=True)"` (no Pi dependencies needed for syntax check)

## Hardware Notes

- Camera: IMX708 on Waveshare CM5 carrier, needs `dtoverlay=imx708,cam0` in `/boot/firmware/config.txt`. Capture at full 2304×1296 for widest FOV, downscale to 640px for LLM.
- Pi 5 venv needs `--system-site-packages` (python3-libcamera is system-only)
- gpiod v2 on Pi Zero 2W: `get_value()` returns bool (False = pressed)
- SSH user: `austingibb` on both Pis
