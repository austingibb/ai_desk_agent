# AI Roommate

An autonomous AI agent that lives on a Raspberry Pi in your room. It watches what's going on through a camera, talks to you through an e-ink display and web chat, and speaks out loud via text-to-speech. It's not a voice assistant you summon — it runs on its own, observing the room and deciding when to chime in.

The real point: keeping you honest about the small daily stuff. Getting up from the desk, drinking water, staying on track with studying instead of drifting. It's the friend who actually remembers what you said you'd do and holds you to it.

## How it works

The system uses a **two-model architecture** split across three devices:

- **DeepSeek on OpenRouter** is the brain. It handles all reasoning, tool calling, conversation, and decisions. It's text-only — it never sees images directly.
- **Gemma 4 31B running locally on llama.cpp** handles vision. A background thread captures photos every ~3 minutes, sends them to Gemma for a text description, and caches the result. When the brain wants to "see" the room, it gets this cached description instantly.
- **Piper TTS** (optional) gives it a voice. The AI's display messages are spoken aloud through a Bluetooth speaker via a local Piper HTTP server.

The brain runs in an autonomous agent loop — there are no timers or hardcoded behaviors. The AI decides what to do and when:

1. Send conversation history + tools to DeepSeek
2. DeepSeek picks an action: check the room, update the display, send a chat message, search the web, wait, or manage notifications
3. Execute the tool calls, feed results back
4. Repeat

When nothing is happening, it idles. When the user interacts (chat message or button press), it wakes up and responds. It manages its own pacing — backing off when ignored, engaging more when in conversation.

## Hardware

| Device | Role |
|--------|------|
| **Pi 5** (or faster) | Orchestrator — runs the agent loop, camera, TTS, web chat server |
| **Pi Zero 2W** | Display server — drives the e-ink screen (SSD1680Z, 122x250) and two GPIO buttons |
| **GPU machine** | Runs llama.cpp serving Gemma 4 31B for vision |

The camera is an IMX708 capturing at full 2304x1296 sensor FOV, downscaled to 640px for the vision model. The e-ink display is small — about 140 characters max — which forces the AI to be concise. Longer thoughts go to the web chat instead.

## Interaction

**E-ink display** — The AI's primary output. Short, punchy messages like texts from a friend. Updated whenever the AI has something to say.

**Web chat** — A password-protected web UI on port 8080. The AI sends longer messages here — real thoughts, stories, detailed replies. The user types back. Optional HTTPS via mkcert.

**Physical buttons** — Two GPIO buttons on the display Pi. Press one to nudge the AI into saying something new, or to approve a proposed notification.

**Voice** — When TTS is enabled, display messages are spoken aloud through Piper. Non-blocking with interrupt support (new speech cuts off old speech).

**Notifications** — The AI can propose recurring reminders (stretch breaks, "it's getting late"). The user approves with a button press or rejects via chat. A scoring system tracks what the user engages with.

## Context and memory

Conversation history persists to disk across restarts. The system auto-compacts at 150 messages, summarizing older messages while keeping the last 30 intact. The AI also has access to Brave Search via MCP for pulling in news, weather, facts, or anything else that sparks a thought.

Motion detection adjusts the vision loop — when the room is still for 5 minutes, it enters a chill mode and stops burning compute on unchanged scenes.

## Configuration

All configuration is via environment variables or a `.env` file. Key settings:

| Variable | Default | What it does |
|----------|---------|-------------|
| `LLM_API_KEY` | _(required)_ | OpenRouter API key |
| `LLM_MODEL` | `deepseek/deepseek-chat` | Brain model |
| `VISION_BASE_URL` | `http://localhost:8081/v1` | llama.cpp server URL |
| `ENABLE_CAMERA` | `1` | Disable camera/vision with `0` |
| `ENABLE_TTS` | `0` | Enable Piper TTS with `1` |
| `CHAT_PASSWORD` | `admin` | Web chat login password |
| `CHAT_USE_HTTPS` | `0` | Enable HTTPS with `1` |
| `VISION_POLL_INTERVAL` | `180` | Seconds between background photo captures |
| `COMPACT_AFTER_N_MESSAGES` | `150` | Message count before compaction triggers |

## Deployment

Both Pis run systemd services. The orchestrator is `ai-eink` on the Pi 5, and the display server is `display-server` on the Pi Zero 2W. Piper TTS runs as a separate `piper-tts` service.

Deploy by pushing to the repo, then pulling and restarting on each Pi:

```bash
# Pi 5
ssh user@<pi5-ip> 'cd ~/ai_desk_agent && git pull && sudo systemctl kill ai-eink; sudo systemctl start ai-eink'

# Pi Zero 2W
ssh user@<pizero-ip> 'cd ~/ai_desk_agent && git pull && sudo systemctl restart display-server'
```

Watch logs with:

```bash
ssh user@<pi5-ip> 'sudo journalctl -u ai-eink -f'
```

## Roadmap

See [improvements/2026-07-10-feature-ideas.md](improvements/2026-07-10-feature-ideas.md) for an evaluation of the project and ten proposed features — durable long-term memory, a generalized habit ledger, a presence timeline, voice input, and more.
