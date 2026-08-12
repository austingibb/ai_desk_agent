# AI Roommate

**An AI that lives in your room and keeps you honest about the small daily stuff.** Getting up from the desk, drinking water, staying on track instead of drifting.

You don't summon it. It watches the room through a camera, runs on its own loop, and decides when to say something. It talks through a web chat and an e-ink display, speaks out loud if you want, tracks your caffeine and pomodoros, and remembers what you said you'd do.

The camera never leaves the house. See [why the eyes are local](#why-the-eyes-are-local).

## Quick start (chat-only)

No Raspberry Pis, no e-ink screen, no wiring. In **chat-only mode** the agent runs on one ordinary machine with the web chat as its only interface. The agent loop, tools, compaction, and notifications all work the same. Messages that would go to the e-ink display go to chat instead, and you approve notifications in chat rather than with physical buttons.

```bash
git clone <this-repo> ai_roommate && cd ai_roommate
pip install -r requirements-chat.txt

export LLM_API_KEY=sk-or-...   # your OpenRouter key
export ENABLE_DISPLAY=0        # no e-ink / buttons, route everything to chat
export ENABLE_CAMERA=0         # no webcam / vision server
export ENABLE_REOLINK=0        # no network security camera

python main.py
```

Open `http://localhost:8080` and log in with `CHAT_PASSWORD` (default `admin`). In the default `smart` approval mode you can answer a notification proposal in plain language. The agent decides whether you approved it, and leaves ambiguous replies pending.

`ENABLE_DISPLAY`, `ENABLE_CAMERA`, and `ENABLE_TTS` toggle independently, so a laptop with a webcam can run camera-on, display-off. Settings can live in a `.env` file instead of the environment. `.env.mac.example` is a working starting point for macOS.

## Local vision (optional)

Room perception always runs against a local vision model, so watching the room continuously never uploads camera frames to a hosted API. The camera layer picks `picamera2` on a Raspberry Pi and OpenCV/AVFoundation on macOS (`pip install -r requirements-mac.txt`). macOS asks for camera permission on the first capture; grant it to whatever terminal or app launches Python. If the wrong webcam is picked, change `CAMERA_DEVICE_INDEX`.

There are two providers.

### `generic`: any OpenAI-compatible vision server

The default. Point `VISION_BASE_URL` at a llama.cpp, mlx-vlm, or similar server and set `VISION_MODEL` to whatever it serves. The agent sends a free-text prompt (the base prompt plus your `requests_for_image_model.md`) and stores the reply as the room description.

```dotenv
ENABLE_CAMERA=1
VISION_PROVIDER=generic
VISION_BASE_URL=http://<your-vision-host>:8080/v1
VISION_MODEL=<model-id-your-server-reports>
```

### `aarg_mlx`: structured Gemma 4 on Apple silicon

This mode replaces the free-text prompt with a fixed image-first prompt and a strict JSON schema, so every frame comes back in the same shape (people present, activity, lighting, notable details) instead of prose. It expects a local [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) server on `127.0.0.1:8090`.

Setting up your own MLX server:

```bash
python3 -m venv ~/.venvs/mlx-vlm
~/.venvs/mlx-vlm/bin/pip install mlx-vlm
~/.venvs/mlx-vlm/bin/pip install huggingface_hub[cli]
~/.venvs/mlx-vlm/bin/hf download mlx-community/gemma-4-12B-it-4bit

~/.venvs/mlx-vlm/bin/python -m mlx_vlm.server \
  --model mlx-community/gemma-4-12B-it-4bit \
  --host 127.0.0.1 --port 8090 \
  --vision-cache-size 1 \
  --thinking-start-token '<|think|>' \
  --thinking-end-token '<channel|>' \
  --max-tokens 1600
```

Confirm it's up with `curl --fail --silent http://127.0.0.1:8090/v1/models`.

Notes on those flags and numbers:

- The weights are about 6.3 GB. A cold request takes roughly 8.5 s and peaks near 7.7 GB of MLX memory, so run vision off your capture and UI threads.
- `<|think|>` and `<channel|>` are Gemma 4's actual reasoning delimiters. mlx-vlm's generic `<think>`/`</think>` defaults don't enforce a thinking budget for this model.
- `--vision-cache-size 1` because unique camera frames get nothing out of caching 20 vision embeddings.
- The server binds to loopback and has no authentication, so don't expose the port.
- Don't keep another large local model resident at the same time on a 32 GB machine.

To point the agent at it: this provider imports `scene.py` (the canonical prompt, JSON schema, and validator) from a companion `aarg_mlx` checkout. Set `VISION_AARG_MLX_DIR` to that directory or put it on `PYTHONPATH`. Startup fails with a clear error if the module isn't found. Without it, use `VISION_PROVIDER=generic` against the same server.

```dotenv
ENABLE_CAMERA=1
VISION_PROVIDER=aarg_mlx
VISION_BASE_URL=http://127.0.0.1:8090/v1
VISION_MODEL=mlx-community/gemma-4-12B-it-4bit
VISION_AARG_MLX_DIR=/path/to/aarg_mlx
VISION_ENABLE_THINKING=1
VISION_THINKING_BUDGET=1024
VISION_MAX_TOKENS=350
```

A shared lock allows only one vision inference at a time, so an on-demand capture can't overlap the background loop.

## How it works

Two models split the work. A hosted brain (DeepSeek V4 Flash on OpenRouter by default) does the reasoning, tool calling, conversation, and decisions. A local vision model does room perception, and nothing else: a background thread captures frames when motion warrants it and caches the description, so the brain's `take_photo` tool returns instantly.

### Why the eyes are local

The split is a privacy boundary, not a performance trick.

A text description and a photograph are not the same risk. "One person at the desk, lamp on, laptop open, room is a mess" is about as identifying as a weather report. The frame it came from is not. That image has your face in it, your apartment, whatever is on your screen, whatever is on your floor, and a timestamp saying you were home.

And it isn't only your call to make. Your roommate walking past, a partner asleep in the background, someone on a video call behind you, a friend who came over for an hour. None of them agreed to be photographed every few minutes and uploaded to a company's servers. The hosted brain's provider isn't really the point. Every hosted model means someone else's hardware, in a jurisdiction you didn't pick, under a retention policy you didn't write.

So the room camera only ever talks to a model running on hardware you own. Only the text description crosses the network. When `LLM_SUPPORTS_IMAGES=1`, an image you deliberately attach in chat is the exception and goes straight to the brain.

When enabled, chat uploads stay only in the newest 30 messages and are never written to `context.json`. After that they're replaced by the post text plus the model's description of them, so old photos don't accumulate in a file on disk.

Everything the vision model is asked, and everything it answers, is logged locally. See [Vision audit trail](#vision-audit-trail).

### The loop

The brain runs an autonomous loop with no timers or hardcoded behaviors:

1. Send conversation history + tool definitions to the brain
2. It picks an action: check the room, update the display, send a chat message, search the web, log a drink or pomodoro, wait, or manage notifications
3. Execute the tool calls, feed results back
4. Repeat

When nothing is happening it idles. A chat message or button press wakes it. It paces itself, backing off when ignored and engaging more during conversation.

Conversation history persists across restarts and auto-compacts at 150 messages, summarizing older ones while keeping the last 30 intact. Brave Search is available through MCP when `ENABLE_WEB_SEARCH=1`. When the room is still for 5 minutes the vision loop enters chill mode and stops describing frames until something moves.

## Vision audit trail

The prompt sent to the vision model lives in its own nested Git repository, separate from this project and with no remote:

```text
requests_for_image_model/
├── .git/
└── requests_for_image_model.md
```

The parent repository ignores the whole directory, so prompt history never lands in normal commits. `update_vision_requests` creates a nested commit and returns its hash. Manual edits are committed automatically before the next vision request, so the recorded hash always matches the prompt actually used.

```bash
git -C requests_for_image_model log --oneline
git -C requests_for_image_model show <commit>:requests_for_image_model.md
```

Successful descriptions are appended to `vision_logs/descriptions.jsonl`, which is also git-ignored. Each entry has the description plus capture and completion times, camera source, provider, model, latency, usage/timings, and the request commit hash. No image bytes.

```bash
tail -f vision_logs/descriptions.jsonl
```

## Configuration

Everything is set with environment variables or a `.env` file.

| Variable | Default | What it does |
|----------|---------|-------------|
| `LLM_API_KEY` | _(required)_ | OpenRouter API key |
| `LLM_MODEL` | `deepseek/deepseek-v4-flash-0731` | Hosted brain model |
| `LLM_SUPPORTS_IMAGES` | `0` | `1` enables chat image uploads for a compatible brain model |
| `VISION_PROVIDER` | `generic` | `generic` free-text request, or `aarg_mlx` structured Gemma vision |
| `VISION_BASE_URL` | provider-specific | Vision server URL |
| `VISION_MODEL` | provider-specific | Vision model id |
| `VISION_AARG_MLX_DIR` | _(unset)_ | Directory containing `scene.py`, required by `aarg_mlx` |
| `VISION_ENABLE_THINKING` | `1` | Enable thinking in `aarg_mlx` perception requests |
| `VISION_THINKING_BUDGET` | `1024` | Thinking-token budget |
| `VISION_POLL_INTERVAL` | `180` | Seconds between background captures |
| `ENABLE_DISPLAY` | `1` | `0` = chat-only mode (no e-ink, no GPIO) |
| `ENABLE_CAMERA` | `1` | `0` disables the camera and all vision tools |
| `ENABLE_REOLINK` | `1` | `0` disables the network security-camera tools |
| `ENABLE_TTS` | `0` | `1` enables Piper TTS |
| `ENABLE_WEB_SEARCH` | `1` | `0` runs without the Brave Search MCP server |
| `MCP_URL` | `http://localhost:8089/mcp` | Brave Search MCP endpoint |
| `ENABLE_STATUS_PUBLISH` | `1` | Publish `{active, drinks}` to S3, needs `STATUS_S3_BUCKET` |
| `NOTIFICATION_APPROVAL_MODE` | `smart` | `smart` interprets chat replies, `legacy` is button-only |
| `CAMERA_BACKEND` | `auto` | `picamera2` on Pi, OpenCV on macOS |
| `CAMERA_DEVICE_INDEX` | `0` | OpenCV webcam index |
| `CHAT_PASSWORD` | `admin` | Web chat login password |
| `CHAT_USE_HTTPS` | `0` | `1` enables HTTPS (`SSL_CERT_FILE` / `SSL_KEY_FILE`) |
| `CHAT_MAX_IMAGES_PER_MESSAGE` | `4` | Attachment count limit when image support is enabled |
| `CHAT_MAX_MEDIA_BYTES` | `20971520` | Decoded attachment bytes per message when enabled (20 MB) |
| `COMPACT_AFTER_N_MESSAGES` | `150` | Message count before compaction triggers |

## Deployment

Both Pis run systemd services: `ai-eink` on the Pi 5, `display-server` on the Pi Zero 2W, and `piper-tts` separately if TTS is enabled. Push to the repo, then pull and restart on each:

```bash
ssh <user>@<pi5-ip> 'cd ~/ai_desk_agent && git pull && sudo systemctl kill ai-eink; sudo systemctl start ai-eink'
ssh <user>@<pizero-ip> 'cd ~/ai_desk_agent && git pull && sudo systemctl restart display-server'
ssh <user>@<pi5-ip> 'sudo journalctl -u ai-eink -f'
```

## Appendix: the full hardware build

Chat-only mode is the whole agent. This is what it looks like with the hardware attached.

| Device | Role |
|--------|------|
| **Pi 5** (or faster) | Orchestrator. Agent loop, camera, TTS, web chat server |
| **Pi Zero 2W** | Display server. E-ink screen (SSD1680Z, 122x250) and two GPIO buttons |
| **Vision machine** | Local OpenAI-compatible vision server (llama.cpp or mlx-vlm) |

The camera is an IMX708 capturing the full 2304x1296 sensor FOV, downscaled before it reaches the vision model. It needs `dtoverlay=imx708,cam0` in `/boot/firmware/config.txt`, and the Pi 5 venv needs `--system-site-packages` because `python3-libcamera` is system-only.

The display holds about 140 characters. That constraint is doing real work: it forces short, punchy messages, like texts from a friend. Longer thoughts go to chat instead.

What each surface adds:

- **E-ink display.** The ambient one. It's just there in your peripheral vision, no notification, no sound.
- **Web chat.** Password-protected UI on port 8080. Longer messages, plus static PNG, JPEG, and WebP uploads when `LLM_SUPPORTS_IMAGES=1`. Optional HTTPS via mkcert.
- **Physical buttons.** Two GPIO buttons. Nudge the AI into saying something, or approve a proposed notification without opening a browser.
- **Voice.** With `ENABLE_TTS=1`, display messages are spoken through a local Piper server. Non-blocking, and new speech interrupts old.
- **Notifications.** The AI proposes recurring reminders like stretch breaks or "it's getting late". In `smart` mode it interprets natural chat replies; in `legacy` mode only a physical button resolves a proposal. A scoring system tracks what you actually engage with.

## License

MIT. See [LICENSE](LICENSE).
