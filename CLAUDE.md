# AI E-Ink Friend

Two Raspberry Pis running an autonomous AI agent that observes the room through a camera, chats with the user, and displays messages on an e-ink screen. The AI uses tool calling (take_photo, update_display, wait, Brave Search MCP) to control its own rhythm.

## Architecture

```
Pi 5 (192.168.0.39) — Orchestrator (main.py)
├── camera.py        → Picamera2 capture → base64 JPEG data URI
├── ai_client.py     → OpenAI-compatible HTTP client (llama.cpp + Gemma 4 31B)
├── context.py       → Message history, timestamps, token counting, compaction
├── mcp_client.py    → Brave Search MCP integration (JSON-RPC over SSE/HTTP)
├── sounds.py        → Non-blocking PulseAudio playback for tool events
└── chat server :8080  → Web UI for user to type messages

Pi Zero 2W (192.168.0.38) — Display Server (display_server.py :5050)
├── display.py       → SSD1680Z e-ink driver (122×250 via SPI)
└── buttons.py       → GPIO button polling (YES=5, NO=6, active LOW)

LLM Server (192.168.0.4:8081)
└── llama.cpp running Gemma 4 31B Q4 with OpenAI-compatible API
```

## Agent Loop (main.py Orchestrator._turn, line 118)

1. Build tools list (core + MCP tools)
2. Send messages + tool definitions to LLM
3. Store assistant response (only if it has tool_calls)
4. Execute each tool call, store results in context
5. After a successful `update_display`, enforce a physical wait (backoff 10→30→90→270→810s)
6. Check compaction (every turn, after tool execution)
7. If no tool calls → idle timeout → nudge message → restart

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | Orchestrator loop, tool execution, chat server, signal handling |
| `config.py` | All constants, system prompt, tool definitions |
| `context.py` | Message store with timestamps, compaction logic |
| `ai_client.py` | LLM HTTP client, Gemma-specific text parsing, compaction summarizer |
| `camera.py` | Picamera2 JPEG capture → base64 data URI |
| `mcp_client.py` | Brave Search MCP client (discovery + tool calls) |
| `display_server.py` | HTTP API for display updates, button state, health checks |
| `display.py` | E-ink hardware driver (PIL text rendering) |
| `buttons.py` | GPIO button reading via gpiod v2 |
| `sounds.py/sounds/` | PulseAudio sound effects for tool events |

## Configuration

All in `config.py`. Key constants:

- `COMPACT_AFTER_N_MESSAGES` (default: 150, env override) — trigger compaction by message count
- `KEEP_LAST_N_MESSAGES` (30) — messages kept after compaction
- `MAX_TOOL_CALLS_PER_TURN` (10) — safety limit per agent turn
- `MIN_PHOTO_INTERVAL` (5s), `MIN_DISPLAY_INTERVAL` (10s) — rate limits
- `BACKOFF_BASE` (10s), `BACKOFF_MAX` (900s) — wait backoff (triples each cycle, resets on interaction)
- `IDLE_TIMEOUT` (60s) — seconds before nudging idle AI
- `LLM_MAX_TOKENS` (2048), `LLM_TIMEOUT` (120s) — LLM parameters

## Context & Compaction

Messages are stored in OpenAI format with a private `_ts` (timestamp) field. `get_messages()` injects human-readable timestamps like `[Wed 14:30:22]` into content before sending to the LLM.

Compaction triggers at `COMPACT_AFTER_N_MESSAGES` (150). It summarizes everything except the system prompt and last `KEEP_LAST_N_MESSAGES` (30) into a single `[Previous context summary: ...]` message. Images are excluded from persistence (`context.json`).

## LLM Quirks (Gemma 4 on llama.cpp)

- Strips control token junk like `<|"|>` from output
- Parses inline tool calls Gemma sometimes emits as text: `<|tool_call>call:wait{seconds:600}<tool_call|>`
- Parses non-JSON argument format `{key:value, ...}`
- Cleans string arguments from control tokens

## The System Prompt

Defines a friendly, chatty persona with three core tools (take_photo, update_display, wait) plus Brave Search MCP tools. Gives the AI a rhythm: think → share thought → wait → update display → wait → repeat. Warns about emoji limitations on e-ink (use text emoticons).

## Hardware Notes

- Camera: Waveshare CM5 carrier needs `dtoverlay=imx708,cam0` in `/boot/firmware/config.txt`
- Pi 5 venv needs `--system-site-packages` (python3-libcamera is system-only)
- gpiod v2 on Pi Zero 2W: `get_value()` returns bool (False = pressed)
- SSH user: `austingibb` on both Pis

## Add New Tool

1. Add tool definition to `TOOL_DEFINITIONS` in `config.py`
2. Add handler in `main.py._execute_tool()`
3. If external tool: add to `mcp_client.py` (Brave Search MCP integration)

## Common Tasks

**Deploy**: Commit + push, then SSH to each Pi and pull + restart services:
```bash
ssh austingibb@192.168.0.39 'cd ~/ai_eink && git pull && sudo systemctl restart ai-eink'
ssh austingibb@192.168.0.38 'cd ~/ai_eink && git pull && sudo systemctl restart display-server'
```

**Watch logs**: `ssh austingibb@192.168.0.39 'sudo journalctl -u ai-eink -f'`

**Test locally**: `python3 -c "import py_compile; py_compile.compile('main.py', doraise=True)"` (no Pi dependencies needed for syntax check)
