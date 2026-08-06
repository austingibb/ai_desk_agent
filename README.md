# AI Roommate

An autonomous AI agent that lives on a Raspberry Pi in your room. It watches what's going on through a camera, talks to you through an e-ink display and web chat, and speaks out loud via text-to-speech. It's not a voice assistant you summon — it runs on its own, observing the room and deciding when to chime in.

The real point: keeping you honest about the small daily stuff. Getting up from the desk, drinking water, staying on track with studying instead of drifting. It's the friend who actually remembers what you said you'd do and holds you to it.

## Run it on your laptop (chat-only)

No Raspberry Pis, no e-ink screen, no wiring. In **chat-only mode** the agent runs on a single ordinary machine and uses the web chat as its only interface. The full agent loop, tools, compaction, and notifications all work the same — the only difference is that messages the AI would put on the e-ink display go to the chat instead, and notification approvals happen in chat (there are no buttons to press).

```bash
git clone <this-repo> ai_roommate && cd ai_roommate
pip install -r requirements-chat.txt

export LLM_API_KEY=sk-or-...   # your OpenRouter key
export ENABLE_DISPLAY=0        # no e-ink / buttons — route everything to chat
export NOTIFICATION_APPROVAL_MODE=smart  # let the agent interpret approval intent
export ENABLE_CAMERA=0         # no webcam / vision server

python main.py
```

Then open `http://localhost:8080`, log in with the chat password (`CHAT_PASSWORD`, default `admin`), and start talking. With smart approval, respond naturally to a notification proposal. The agent interprets whether you approved or rejected it, and leaves ambiguous or unrelated replies pending.

`ENABLE_DISPLAY`, `ENABLE_CAMERA`, and `ENABLE_TTS` toggle independently, so a laptop with a webcam and a reachable vision server can run camera-on, display-off with `ENABLE_CAMERA=1` (use `requirements-mac.txt` on macOS). Set `ENABLE_REOLINK=0` unless you have the security camera on your network.

## Mac mobile mode with AARG MLX vision

The camera layer auto-selects `picamera2` on a Raspberry Pi and OpenCV/AVFoundation
on macOS. Pi deployments retain the existing generic llama.cpp vision request;
Mac mobile mode can opt into the structured Gemma 4 service in `aarg_mlx`.

Start Gemma once and confirm its health:

```bash
cd /Users/austingibbons/tools/aarg_mlx
bin/aarg-mlx stop
bin/aarg-vision start
curl --fail --silent http://127.0.0.1:8090/v1/models
```

Install and configure AI Roommate:

```bash
cd /Users/austingibbons/tools/ai_roommate
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-mac.txt
```

Copy `.env.mac.example` to `.env`, then add your OpenRouter key and choose a
chat password. Its relevant vision settings are:

```dotenv
LLM_API_KEY=sk-or-...
ENABLE_DISPLAY=0
NOTIFICATION_APPROVAL_MODE=smart
ENABLE_CAMERA=1
ENABLE_REOLINK=0
ENABLE_TTS=0
ENABLE_STATUS_PUBLISH=0
ENABLE_WEB_SEARCH=0
CAMERA_BACKEND=auto
CAMERA_DEVICE_INDEX=0
CAMERA_ROTATION=0
VISION_PROVIDER=aarg_mlx
VISION_BASE_URL=http://127.0.0.1:8090/v1
VISION_MODEL=mlx-community/gemma-4-12B-it-4bit
VISION_AARG_MLX_DIR=/Users/austingibbons/tools/aarg_mlx
VISION_ENABLE_THINKING=1
VISION_THINKING_BUDGET=1024
VISION_MAX_TOKENS=350
```

Then run:

```bash
python main.py
```

macOS will ask for camera permission the first time. Grant access to the terminal
or application that launches Python. If the wrong webcam is selected, change
`CAMERA_DEVICE_INDEX`. Copy an existing `context.json` into this directory while
the agent is stopped to resume its conversation history.

In AARG mode, each perception request uses the canonical image-first prompt and
strict scene schema. Thinking is enabled explicitly with Gemma 4's required
`<|think|>` / `<channel|>` delimiters. A shared lock permits only one active
vision inference even if an on-demand capture overlaps the background loop.

## How it works

The system uses a **two-model architecture** split across three devices:

- **MiniMax M3 on OpenRouter** is the brain. It handles reasoning, tool calling, conversation, decisions, and images/GIFs posted through web chat.
- A **local vision model** handles room perception. The Pi path supports its existing OpenAI-compatible llama.cpp server; Mac mobile mode supports Gemma 4 12B through AARG MLX. A background thread captures photos when motion warrants it and caches the resulting description.
- **Piper TTS** (optional) gives it a voice. The AI's display messages are spoken aloud through a Bluetooth speaker via a local Piper HTTP server.

The local room-camera path stays separate so continuous monitoring does not upload
camera frames to OpenRouter. User-posted PNG, JPEG, WebP, and GIF files go directly
to MiniMax M3 as multimodal message content. Raw uploads remain only in the latest
30 conversation messages and are never written to `context.json`; after that they
are replaced with the post text and MiniMax's contemporaneous description before
normal context compaction.

The brain runs in an autonomous agent loop — there are no timers or hardcoded behaviors. The AI decides what to do and when:

1. Send conversation history + tools to MiniMax M3
2. MiniMax M3 picks an action: check the room, update the display, send a chat message, search the web, wait, or manage notifications
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

**Web chat** — A password-protected web UI on port 8080. The AI sends longer messages here — real thoughts, stories, detailed replies. The user can type, paste, select, or drag in PNG, JPEG, WebP, and animated GIF files. Optional HTTPS via mkcert.

**Physical buttons** — Two GPIO buttons on the display Pi. Press one to nudge the AI into saying something new, or to approve a proposed notification.

**Voice** — When TTS is enabled, display messages are spoken aloud through Piper. Non-blocking with interrupt support (new speech cuts off old speech).

**Notifications** — The AI can propose recurring reminders (stretch breaks, "it's getting late"). In the default `smart` approval mode, the agent interprets natural chat responses and explicitly resolves the proposal; physical button presses still approve when the display hardware is enabled. In `legacy` mode, only the physical button resolves a proposal. A scoring system tracks what the user engages with.

## Context and memory

Conversation history persists to disk across restarts. The system auto-compacts at 150 messages, summarizing older messages while keeping the last 30 intact. When enabled, Brave Search is available through MCP for pulling in news, weather, and facts.

Motion detection adjusts the vision loop — when the room is still for 5 minutes, it enters a chill mode and stops burning compute on unchanged scenes.

## Local vision audit trail

Vision requests have their own local Git repository:

```text
requests_for_image_model/
├── .git/
└── requests_for_image_model.md
```

This nested repository is independent from the project repository. It contains
only the requests file, has its own commits and hashes, and has no remote. The
parent repository ignores the entire directory, so its prompt history is never
included in normal commits or pushes.

On first startup, an existing root-level `requests_for_image_model.md` is copied
into the nested repository, committed, and then removed from the old location.
Calls to `update_vision_requests` create a new nested commit and return its full
hash. Manual edits are committed automatically before the next vision request,
ensuring the hash and prompt used for inference match.

Inspect or restore the local history with ordinary Git commands:

```bash
git -C requests_for_image_model log --oneline
git -C requests_for_image_model show <commit>:requests_for_image_model.md
git -C requests_for_image_model checkout <commit> -- requests_for_image_model.md
git -C requests_for_image_model status
```

Successful vision descriptions are written separately to:

```text
vision_logs/descriptions.jsonl
```

Watch new descriptions as they arrive:

```bash
tail -f vision_logs/descriptions.jsonl
```

Each JSONL entry contains only the description and vision audit metadata:
capture/completion times, camera source, provider, model, latency, usage/timings,
and the exact nested request commit hash. No image bytes are written to this log.
The entire `vision_logs/` directory is ignored by the project repository.

## Configuration

All configuration is via environment variables or a `.env` file. Key settings:

| Variable | Default | What it does |
|----------|---------|-------------|
| `LLM_API_KEY` | _(required)_ | OpenRouter API key |
| `LLM_MODEL` | `minimax/minimax-m3` | Multimodal brain model on OpenRouter |
| `VISION_PROVIDER` | `generic` | `generic` for the existing Pi request or `aarg_mlx` for structured Gemma vision |
| `VISION_BASE_URL` | provider-specific | Vision server URL (`http://127.0.0.1:8090/v1` in AARG mode) |
| `VISION_ENABLE_THINKING` | `1` | Enable Gemma thinking in AARG perception requests |
| `VISION_THINKING_BUDGET` | `1024` | Thinking-token budget for AARG perception |
| `VISION_REQUESTS_REPO_DIR` | `requests_for_image_model` | Local nested Git repository for the image-model requests file |
| `VISION_DESCRIPTION_LOG_FILE` | `vision_logs/descriptions.jsonl` | Local JSONL log containing successful vision descriptions |
| `ENABLE_WEB_SEARCH` | `1` | Initialize Brave Search MCP tools; set `0` to run without the MCP server |
| `CAMERA_BACKEND` | `auto` | Auto-select `picamera2` on Pi or OpenCV on macOS |
| `CAMERA_DEVICE_INDEX` | `0` | OpenCV webcam index |
| `ENABLE_DISPLAY` | `1` | Disable the e-ink display + GPIO buttons with `0` (chat-only mode) |
| `NOTIFICATION_APPROVAL_MODE` | `smart` | `smart` lets the agent interpret chat approval/rejection; `legacy` is physical-button-only |
| `ENABLE_CAMERA` | `1` | Disable camera/vision with `0` |
| `ENABLE_TTS` | `0` | Enable Piper TTS with `1` |
| `CHAT_PASSWORD` | `admin` | Web chat login password |
| `CHAT_USE_HTTPS` | `0` | Enable HTTPS with `1` |
| `CHAT_MAX_IMAGES_PER_MESSAGE` | `4` | Maximum image/GIF attachments in one chat message |
| `CHAT_MAX_MEDIA_BYTES` | `20971520` | Maximum decoded attachment bytes per message (20 MB total) |
| `CHAT_TAKEOVER_SECONDS` | `15` | Duration of transient chat notices |
| `CHAT_SSE_MAX_STREAMS` | `8` | Maximum concurrent chat event streams |
| `CHAT_SSE_HEARTBEAT_SECONDS` | `15` | Seconds between event-stream heartbeat comments |
| `CHAT_SSE_IDLE_SECONDS` | `300` | Close an event stream after this many seconds without a state or transcript change |
| `VISION_POLL_INTERVAL` | `180` | Seconds between background photo captures |
| `COMPACT_AFTER_N_MESSAGES` | `150` | Message count before compaction triggers |

## Contributor note: adding tools

Any commit that adds an agent tool must add its plain-English mode-indicator label
to `TOOL_LABELS` in `main.py` in the same commit. Unmapped tools deliberately show
their exact raw name in the UI so missing labels remain visible.

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
