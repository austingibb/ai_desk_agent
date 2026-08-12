# AI Roommate

An autonomous agent that watches a room through a camera, talks through a web chat
and an e-ink display, and decides on its own when to speak. It tracks caffeine and
pomodoros, proposes recurring reminders, and searches the web.

Two models split the work, and the split is a privacy boundary, not a performance
one. A hosted brain (MiniMax M3 on OpenRouter) does reasoning and tool calling. A
local vision model does room perception, and only its text descriptions ever leave
the machine. Camera frames from the room never reach a hosted API. Anything that
weakens that boundary is a design regression, not an optimization.

## Architecture

```
Orchestrator (main.py): Pi 5, or any single machine in chat-only mode
├── config.py         → constants, system prompt, tool definitions, feature flags
├── ai_client.py      → AIClient (brain/OpenRouter) + VisionClient (local, 2 providers)
├── context.py        → message store, timestamps, compaction, pairing repair
├── chat_media.py     → user image validation + multimodal content construction
├── camera.py         → capture backend: picamera2 on Pi, OpenCV/AVFoundation on macOS
├── scene_change.py   → motion detection via phase correlation (Pillow + numpy)
├── vision_history.py → nested-git prompt history + JSONL description log
├── reolink.py        → optional network security camera (snapshot, IR, spotlight)
├── mcp_client.py     → Brave Search MCP (JSON-RPC over SSE/HTTP)
├── notifications.py  → proposals, approval/rejection, decay scoring, reviews
├── caffeine.py       → DrinkStore, append-only drinks.json, 30-day retention
├── pomodoro.py       → PomodoroStore, append-only pomodoros.json, never pruned
├── presence.py       → ActiveTracker, at-desk boolean from motion/chat/buttons
├── status_publisher.py → uploads {active, drinks} to public S3 for aarg.dev
├── tts.py            → Piper HTTP speech, fire-and-forget with interrupt
├── sounds.py         → non-blocking PulseAudio effects for tool events
├── logger.py         → timestamped info() to journal + optional VERBOSE_LOG file
├── static/           → chat UI assets (chat.html, chat.css, chat.mjs, chat_model.mjs)
└── chat server :8080 → web UI, SSE stream, queue controls

Display server (display_server.py :5050): Pi Zero 2W
├── display.py        → SSD1680Z e-ink driver, 122×250 over SPI, PIL text rendering
└── buttons.py        → GPIO polling via gpiod v2 (YES=5, NO=6, active LOW)

Local vision server: separate machine or the same one
└── llama.cpp, mlx-vlm, or anything OpenAI-compatible
```

`harvest_memory.py` is a standalone offline script, not part of the running agent.
It chunks `context.json` and asks the brain to report mistakes and improvements.

## Two models

- **Brain** (`LLM_MODEL`, default `deepseek/deepseek-v4-flash-0731` on
  OpenRouter). Reasoning, tool calling, display decisions, notification
  management, and compaction. `LLM_SUPPORTS_IMAGES` declares whether it can
  directly understand images posted in chat; the default model cannot.
- **Vision** (`VISION_MODEL`, local). Describes room-camera frames. Nothing else.

The brain reaches the room through `take_photo`, which returns the cached
description from the background vision thread with no round trip. `capture_photo`
takes a fresh frame and blocks on the vision model (up to `VISION_TIMEOUT`); it
exists for moments that genuinely need current information and should stay rare.

When `LLM_SUPPORTS_IMAGES=1`, user-posted PNG/JPEG/WebP files skip the local
model and go to the brain as `image_url` content. Raw media lives only in the
newest `KEEP_LAST_N_MESSAGES` messages and is never written to `context.json`;
`demote_old_images()` replaces older uploads with the post text plus the brain's
contemporaneous description before compaction ever sees them. With the flag off,
the server rejects image payloads and the chat UI hides its attachment control.

`VisionClient` supports two providers:

- `generic`: free-text prompt (`VISION_PROMPT_BASE` plus the requests file) to any
  OpenAI-compatible server. This is the Pi path.
- `aarg_mlx`: imports `scene.py` from a companion `aarg_mlx` checkout
  (`VISION_AARG_MLX_DIR` or `PYTHONPATH`) for a canonical image-first prompt and a
  strict JSON schema, so every frame returns the same structure. Gemma 4 on
  mlx-vlm. Raises at startup if the module is missing.

An `_inference_lock` allows one vision inference at a time, so an on-demand
capture cannot overlap the background loop.

## Agent loop (`Orchestrator._turn`)

1. Drain the chat queue into context (thread-safe via `ctx_lock`)
2. `demote_old_images()`, then `_repair_pairing()` for OpenRouter format
3. Build the tool list (core + MCP), append the turn reminder
4. Estimate tokens; compact if over `LLM_ESTIMATED_MAX_TOKENS`
5. Send messages + tools to the brain
6. Store the assistant response
7. Execute each tool call, store results, update the UI mode indicator
8. After `update_display`, enforce a wait unless the brain already called `wait`
9. Check compaction and summary merging after tool results
10. No tool calls → idle timeout → nudge → restart

Context-overflow errors from the API also trigger compaction and a retry. LLM
errors back off (`BACKOFF_BASE` tripling to `BACKOFF_MAX`), and the backoff resets
on user interaction.

## Background vision loop

A daemon thread (`_start_vision_loop`) with two tiers. Motion drives everything;
there is no unconditional capture timer.

1. Every `MOTION_POLL_INTERVAL` (2s), grab a cheap lores frame and run
   `scene_change.check()` (phase correlation, so panning doesn't read as motion)
2. Motion touches `presence` and, in `chill` mode, immediately captures and
   notifies the agent. In `active` mode it captures only if `VISION_POLL_INTERVAL`
   has elapsed since the last one, so the interval is a cooldown, not a schedule
3. `_do_vision_capture()` describes via `VisionClient.describe()`, retrying up to
   3 times on empty output (the vision model intermittently returns nothing)
4. Cache into `self.latest_scene` under `scene_lock`
5. Append to the description log, save a debug JPEG (24h rolling window)
6. `CHILL_TIMEOUT` (300s) with no motion drops back to `chill`, where no frames are
   described at all until something moves

## Key files

| File | Purpose |
|------|---------|
| `main.py` | Orchestrator, tool dispatch, vision thread, chat server, signals |
| `config.py` | Constants, `build_system_prompt()`, `get_tool_definitions()`, flags |
| `context.py` | Message store, compaction, summary merging, `_repair_pairing()` |
| `ai_client.py` | `AIClient` (brain) + `VisionClient` (generic and aarg_mlx providers) |
| `chat_media.py` | Validates posted media, builds multimodal messages, authed previews |
| `camera.py` | Backend selection, capture, downscale, JPEG encode |
| `scene_change.py` | Phase-correlation motion detection |
| `vision_history.py` | `VisionRequestHistory` (nested git) + `VisionDescriptionLog` (JSONL) |
| `reolink.py` | `ReoLinkCamera`. Snapshot, IR, spotlight over HTTP |
| `notifications.py` | Proposals, approval/rejection, decay scoring, review summaries |
| `caffeine.py` | `DrinkStore`. drinks.json, pruned to 30 days |
| `pomodoro.py` | `PomodoroStore`. pomodoros.json, local-timezone stats and streaks |
| `presence.py` | `ActiveTracker`. At-desk boolean, debounced |
| `status_publisher.py` | Daemon thread uploading the public status feed to S3 |
| `harvest_memory.py` | Offline context analysis, run by hand |
| `setup-aws.sh` | One-time bucket, public-read policy, CORS, scoped IAM user |
| `display_server.py` | HTTP API for display updates, button state, health |
| `display.py` | E-ink driver |
| `buttons.py` | GPIO reading via gpiod v2 |

Runtime state lives in `context.json`, `notifications.json`, `drinks.json`,
`pomodoros.json`, `user_data/`, `debug_images/`, `vision_logs/`, and
`requests_for_image_model/`. All are git-ignored.

## Tools

Defined in `config.py:TOOL_DEFINITIONS`, dispatched in `main.py._dispatch_tool()`.

| Group | Tools |
|-------|-------|
| Vision | `take_photo`, `capture_photo`, `update_vision_requests` |
| Reolink | `take_reolink_photo`, `flash_ir_light`, `flash_camera_light` |
| Output | `update_display`, `send_chat_message` |
| Pacing | `wait` |
| Notifications | `propose_notification`, `resolve_notification_proposal`, `schedule_notification`, `delete_notification` |
| Caffeine | `log_drink`, `list_drinks`, `edit_drink` |
| Pomodoro | `log_pomodoro`, `list_pomodoros`, `edit_pomodoro`, `pomodoro_stats`, `enter_pomodoro_mode`, `exit_pomodoro_mode` |
| Search (MCP) | `brave_web_search`, `brave_local_search`, `brave_image_search`, `brave_video_search`, `brave_news_search`, `brave_summarizer` |

`get_tool_definitions()` filters by flag: `CAMERA_TOOL_NAMES` drop out when
`ENABLE_CAMERA=0`, `REOLINK_TOOL_NAMES` when `ENABLE_REOLINK=0`, and
`SMART_NOTIFICATION_TOOL_NAMES` outside `smart` approval mode. `build_system_prompt()`
drops the matching prose so the prompt never describes a tool that isn't offered.

The agent converts drink names to mg using a reference table in the system prompt.
Pomodoro mode takes over the e-ink with a focus screen; while it is on,
`update_display` and the `send_chat_message` preview deliberately skip the display
write and say so in their result.

## Feature flags

`ENABLE_DISPLAY`, `ENABLE_CAMERA`, `ENABLE_TTS`, `ENABLE_REOLINK`,
`ENABLE_WEB_SEARCH`, and `ENABLE_STATUS_PUBLISH` are independent, so a laptop can
run camera-on/display-off, or chat-only with everything else off.

`ENABLE_DISPLAY=0` (chat-only mode) changes:

- `update_display` and the chat preview skip the display-server HTTP call. The text
  still reaches the user because the chat UI renders those tool calls.
- No button polling in `_tool_wait` / `_idle_wait`; `_display_error` is a no-op.
- `build_system_prompt()` swaps e-ink and button wording for chat wording.
- Notification approval follows `NOTIFICATION_APPROVAL_MODE`. `smart` (default)
  sends natural-language replies to the agent and exposes
  `resolve_notification_proposal`; `legacy` keeps the fixed keyword matcher.
  Button approval is unchanged when the display is on.

`picamera2` and `numpy` imports are lazy, so `python main.py` runs on a laptop
without Pi libraries. Deps: `requirements.txt` (Pi), `requirements-chat.txt`
(chat-only), `requirements-mac.txt` (chat + webcam), `requirements-display.txt`
(Pi Zero).

## Context and compaction

Messages are stored in OpenAI format with a `_ts` field and a stable id.
`get_messages()` injects human-readable timestamps like `[Wed 14:30:22]` before
sending. `total_tokens()` estimates at `TOKEN_ESTIMATE_DIVISOR` chars per token.

Compaction fires at `COMPACT_AFTER_N_MESSAGES` (150) or on token overflow. It
summarizes everything except the system prompt and the last `KEEP_LAST_N_MESSAGES`
(30) into a `[Previous context summary: ...]` message, using `_find_safe_end()` to
avoid splitting an assistant message from its tool results. Once summaries pass
`MERGE_SUMMARIES_AFTER` (20), `check_merge_summaries()` collapses them toward
`MERGE_SUMMARIES_TARGET` (15). Both use the brain model.

### OpenRouter message pairing

OpenRouter requires that an assistant message with `tool_calls` is immediately
followed by matching `tool` results. `_repair_pairing()` fixes three violations:

1. Orphan tool messages with no matching assistant
2. Non-tool messages sandwiched between an assistant and its tool results
3. Unfulfilled `tool_calls`, trimmed from the assistant's list

Chat messages are queued in `chat_queue` and drained only at safe points, which is
what keeps pairing intact. Do not append to context from the HTTP thread.

## Chat server

Web UI on `:8080`, password login with a session cookie, optional HTTPS.

| Route | Behavior |
|-------|----------|
| `GET /` | Chat HTML, or the login page |
| `GET /login`, `POST /login` | Password form; sets a 32-byte `HttpOnly` token, `CHAT_SESSION_DAYS` |
| `GET /chat` | Last 50 filtered messages as JSON, deduped between context and queue |
| `POST /chat` | Text + capability-gated base64 attachments; validates, queues, signals the loop |
| `GET /chat/events` | SSE stream of UI state and transcript revisions |
| `GET /chat/media/<sha256>` | Authenticated preview served from live context |
| `PATCH /chat/queue/<id>` | Edit a queued message before the agent drains it |
| `DELETE /chat/queue/<id>` | Undo a queued message |
| `GET /static/<asset>` | Chat assets, cache-busted by `_chat_asset_version()` |

`ChatUIState` holds the agent mode indicator (`thinking`, plus the `TOOL_LABELS`
string for the running tool) and a revision counter. SSE clients long-poll on the
revision with heartbeats; limits are `CHAT_SSE_MAX_STREAMS`,
`CHAT_SSE_HEARTBEAT_SECONDS`, `CHAT_SSE_IDLE_SECONDS`.

The client appends new messages rather than replacing the DOM, tracks what it has
rendered by content signature, renders uploaded media, capability-gates file
selection/paste/drag-drop, and auto-scrolls only when already at the bottom.

## Vision request history

`requests_for_image_model/` is a nested Git repository with its own commits and no
remote. The parent repo ignores the whole directory, so prompt history never lands
in normal commits. `update_vision_requests` writes atomically and commits, returning
the full hash. Manual edits are committed automatically before the next vision
request, so the recorded hash always matches the prompt actually used.

Successful descriptions append to `vision_logs/descriptions.jsonl`: description,
capture and completion times, camera source, provider, model, latency,
usage/timings, and the request commit hash. No image bytes, ever.

## Notifications

The agent proposes recurring reminders and the user approves them by button or, in
`smart` mode, in plain language. `notifications.py` scores what the user engages
with and decays what they ignore. Pacing: `REVIEW_INTERVAL` (30 min),
`MAX_PROPOSAL_INTERVAL` (2h), `MAX_FIRINGS_PER_HOUR` (1),
`CATEGORY_COOLDOWN_REVIEWS` (3).

## Public status feed

`status_publisher.py` uploads `{"active": bool, "drinks": [{"t": epoch_ms, "mg": int}]}`
to S3 every `STATUS_PUBLISH_INTERVAL` (45s). aarg.dev depends on this exact shape:
raw events, no decay math, 30-day retention, never a future timestamp. `active` is
false after `ACTIVE_WINDOW_SECONDS` (300) with no motion, chat, or button press.

This feed is public. AWS credentials live only in the Pi 5's `.env`
(`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`); boto3 reads
them from the environment. Infrastructure is bootstrapped once with `setup-aws.sh`.

## Configuration

Everything is in `config.py`, all env-overridable. Defaults worth knowing:

| Group | Constants |
|-------|-----------|
| Network | `DISPLAY_SERVER_URL`, `MCP_URL` (Brave Search MCP) |
| Brain | `LLM_BASE_URL` (OpenRouter), `LLM_API_KEY`, `LLM_MODEL`, `LLM_SUPPORTS_IMAGES`, `LLM_MAX_TOKENS` (2048), `LLM_MAX_TOKENS_COMPACT` (64000), `LLM_TIMEOUT` (120) |
| Vision | `VISION_PROVIDER` (`generic`), `VISION_BASE_URL`, `VISION_MODEL`, `VISION_AARG_MLX_DIR`, `VISION_ENABLE_THINKING`, `VISION_THINKING_BUDGET` (1024), `VISION_MAX_TOKENS`, `VISION_TIMEOUT` (120), `VISION_POLL_INTERVAL` (180), `MOTION_POLL_INTERVAL` (2.0), `CHILL_TIMEOUT` (300) |
| Context | `COMPACT_AFTER_N_MESSAGES` (150), `KEEP_LAST_N_MESSAGES` (30), `MAX_CONTEXT_TOKENS` (64000), `MERGE_SUMMARIES_AFTER` (20), `MERGE_SUMMARIES_TARGET` (15) |
| Pacing | `BACKOFF_BASE` (10), `BACKOFF_MAX` (900), `MAX_TOOL_CALLS_PER_TURN` (10), `MIN_DISPLAY_INTERVAL` (10), `MIN_WAIT_SECONDS` (10), `MAX_WAIT_SECONDS` (1800), `IDLE_TIMEOUT` (60) |
| Chat | `CHAT_PASSWORD` (`admin`), `CHAT_SESSION_DAYS` (7), `CHAT_USE_HTTPS`, `SSL_CERT_FILE`, `SSL_KEY_FILE`, `CHAT_MAX_IMAGES_PER_MESSAGE` (4), `CHAT_MAX_MEDIA_BYTES` (20 MB), `CHAT_TAKEOVER_SECONDS` (15) |
| Camera | `CAMERA_BACKEND` (`auto`), `CAMERA_DEVICE_INDEX`, `CAMERA_WIDTH` (2304), `CAMERA_HEIGHT` (1296), `CAMERA_IMAGE_MAX_WIDTH`, `CAMERA_ROTATION`, `JPEG_QUALITY` |
| Pomodoro | `POMODORO_WORK_MINUTES` (25), `POMODORO_BREAK_MINUTES` (5), `POMODORO_IDLE_EXIT_SECONDS` (1800) |

Several vision and camera defaults switch on `VISION_PROVIDER`: `aarg_mlx` gets a
1024px long edge, JPEG quality 88, and 350 max tokens, against 640px/50/2048 for
`generic`. Change both branches when touching either.

`user_data/rules.md`, if present, is read into the system prompt. It is user-owned
and git-ignored. Never write to it from a tool handler.

## Add a new tool

1. Add the definition to `TOOL_DEFINITIONS` in `config.py`
2. Add a `_tool_*` handler and wire it into `main.py._dispatch_tool()`
3. Add a plain-English entry to `TOOL_LABELS` in `main.py`, same commit
4. If it belongs to a flag group, add the name to `CAMERA_TOOL_NAMES`,
   `REOLINK_TOOL_NAMES`, or `SMART_NOTIFICATION_TOOL_NAMES` in `config.py`, and
   gate the matching system-prompt prose
5. If it is an external tool, add it to `mcp_client.py`

Step 3 is required, not optional. Every agent tool gets a `TOOL_LABELS` entry in
the same commit that adds the tool. A tool with no label shows its raw function
name in the chat UI's mode indicator. That is intentional: the gap stays visible
instead of falling back to something generic that hides it.

## Common tasks

**Commits**: no AI attribution. No `Co-Authored-By: Claude` trailer, no "Generated
with Claude Code" line.

**Syntax check** (no Pi dependencies needed):

```bash
python3 -c "import py_compile; py_compile.compile('main.py', doraise=True)"
```

**Tests**: stdlib `unittest`, no pytest.

```bash
python3 -m unittest discover -p 'test_*.py'
node --test test_chat_ui.mjs
```

**Deploy**: commit and push, then pull and restart on each Pi.

```bash
ssh <user>@<pi5-ip> 'cd ~/ai_desk_agent && git pull && sudo systemctl kill ai-eink; sudo systemctl start ai-eink'
ssh <user>@<pizero-ip> 'cd ~/ai_desk_agent && git pull && sudo systemctl restart display-server'
ssh <user>@<pi5-ip> 'sudo journalctl -u ai-eink -f'
```

## Hardware notes

- Camera: IMX708 on a Waveshare CM5 carrier, needs `dtoverlay=imx708,cam0` in
  `/boot/firmware/config.txt`. Capture the full 2304×1296 for the widest FOV.
- The Pi 5 venv needs `--system-site-packages`; `python3-libcamera` is system-only.
- gpiod v2 on the Pi Zero 2W: `get_value()` returns a bool, and `False` means
  pressed.
- The e-ink display holds roughly 140 characters. That limit is why the agent has
  both `update_display` and `send_chat_message`.
