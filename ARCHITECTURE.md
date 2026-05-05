# AI E-Ink Roommate — Project State

## What it does

Two Raspberry Pis create an AI roommate experience:
- **Pi 5 CM5 (.39)** with Camera Module 3: takes photos every 2 minutes, sends to Gemma4 AI
- **Pi Zero 2W (.38)** with SSD1680Z e-ink display (122×250) + 2 buttons: shows AI messages, accepts YES/NO responses
- **Gemma 4 31B Q4** running via llama.cpp at `192.168.0.4:8081/v1` — multimodal, supports base64 images + reasoning_content

The AI silently observes the room every 2 minutes (photo + internal reasoning). It can update the e-ink display with a message — choosing its own timing. After a statement, AI controls cooldown (can speak again anytime). After a question, 30-min minimum cooldown, but user YES/NO button response immediately resets it.

## Architecture

```
Pi 5 (.39) — Orchestrator (main.py)               Pi Zero 2W (.38) — Display Server (display_server.py)
├─ camera.py      → picamera2 + PIL JPEG          ├─ display.py      → SSD1680Z via SPI (CS=CE0, DC=22, RST=27, BUSY=17)
├─ ai_client.py   → Gemma4 HTTP + streaming       ├─ buttons.py      → gpiod polling (GPIO 5=YES, 6=NO, pull-up, LOW=pressed)
├─ context.py     → token counting + compaction   └─ HTTP on :5050
└─ main.py        → orchestrator loop                  POST /display {"text":"...", "question":"..."}
                                                       GET /buttons/check  → {"pressed":bool, "button":"YES"/"NO"}
                                                       GET /buttons/wait?timeout=N → {"response":"YES"/"NO"/null}
                                                       GET /health
```

### Data flow per 2-minute cycle:
1. Camera captures JPEG (640×480), PIL encodes, base64 data URI
2. Context built: system prompt + compacted prior windows + current window observations (text only) + ONE photo image
3. Sent to Gemma4 via streaming `/v1/chat/completions` — tokens streamed in real-time to journald
4. AI's reasoning_content (thoughts) captured, content parsed for `DISPLAY:`/`MESSAGE:`/`QUESTION:` directives
5. If AI signals display update AND cooldown passed → POST to .38:5050/display
6. If question asked → wait for button, response fed back to context, cooldown resets

### Context management:
- Previous windows compacted into summaries (~80k token limit)
- Within a window: reasoning stored as text (photos NOT kept — only latest photo in context)
- Compaction triggered at 70% of max tokens, uses Gemma4 to summarize

## Files

| File | Runs on | Purpose |
|------|---------|---------|
| `main.py` | Pi 5 | Orchestrator: photo loop, button check, display decisions, state machine |
| `camera.py` | Pi 5 | Picamera2 capture → PIL JPEG → base64 URI |
| `ai_client.py` | Pi 5 | Gemma4 HTTP client with SSE streaming, response parsing |
| `context.py` | Pi 5 | Message store, token estimation, auto-compaction |
| `config.py` | both | ALL constants: URLs, pins, timing, prompts |
| `display_server.py` | Pi Zero 2W | HTTP API for display + buttons |
| `display.py` | Pi Zero 2W | SSD1680Z driver (Adafruit epd), PIL text rendering |
| `buttons.py` | Pi Zero 2W | gpiod v2 button polling (compare get_value()→bool, False=pressed) |
| `requirements.txt` | Pi 5 | picamera2, requests, Pillow |
| `requirements-display.txt` | Pi Zero 2W | adafruit-circuitpython-epd, adafruit-blinka, gpiod, Pillow |
| `setup.sh` | Pi 5 | apt-get, venv --system-site-packages, pip install, systemd |
| `setup-display.sh` | Pi Zero 2W | Enables SPI, apt-get, venv, systemd |
| `ai-eink.service` | Pi 5 | systemd service, PYTHONUNBUFFERED=1, auto-restart |
| `display-server.service` | Pi Zero 2W | systemd service, auto-restart |

## Deploy workflow

```bash
# Local: commit + push
git add -A && git commit -m "..." && git push

# Pi 5 (.39): pull + restart
ssh austingibb@192.168.0.39 'cd ~/ai_eink && git pull && sudo systemctl restart ai-eink'

# Pi Zero 2W (.38): pull + restart
ssh austingibb@192.168.0.38 'cd ~/ai_eink && git pull && sudo systemctl restart display-server'

# Watch live logs (AI thoughts in real-time)
ssh austingibb@192.168.0.39 'sudo journalctl -u ai-eink -f'
```

## Important gotchas

1. **SSH**: User is `austingibb` on both Pis. Use `~/.ssh/id_rsa`. Flaky — use flags: `-o UserKnownHostsFile=/dev/null -o GSSAPIAuthentication=no -o CheckHostIP=no`
2. **gpiod v2 on Pi Zero 2W**: `get_value()` returns `bool` (True=HIGH/unpressed, False=LOW/pressed), NOT enum. Compare with `not value`.
3. **gpiod v2 on Pi 5**: `request_lines("/dev/gpiochip4", config_dict, consumer="...")` — no chip.get_line()
4. **Camera on Waveshare CM5 carrier**: `camera_auto_detect=1` + `dtoverlay=imx708,cam0` in `/boot/firmware/config.txt`. WITHOUT the overlay, camera won't detect.
5. **Pi 5 venv**: Must use `--system-site-packages` because python3-libcamera is system-only.
6. **Pi 5 system deps**: `libcap-dev` needed for picamera2's python-prctl.
7. **Streaming**: `reasoning_content` comes in `delta` during SSE streaming. Content tokens may also appear (structured output like `*   DISPLAY:...`).
8. **Parsing**: `DISPLAY:` can mean either `yes`/`no` (reasoning) or actual text (consolidation). Parser checks both cases.
9. **Display format**: Text auto-wraps via PIL. Questions shown as `[YES] <question> [NO]` at bottom.
10. **Button press during normal operation**: Polled via HTTP `/buttons/check` every 4 seconds in main loop → triggers `_force_display_update()`.

## Current state

- ✅ Camera captures, AI reasons with streaming
- ✅ Context builds with observations, compacts properly
- ✅ Display server responds, buttons work (simple polling)
- ✅ Display updates work (text + questions sent to e-ink)
- ⚠️ Need to verify: parsed display text + question from consolidation actually reaching display (was getting `...` previously, fixed but not yet verified with latest deploy)
- ⚠️ Button response → AI follow-up not yet tested end-to-end

## Observing live

```bash
ssh austingibb@192.168.0.39 'sudo journalctl -u ai-eink -f'
```

Shows: `[PROMPT msg N]` (text sent to AI), streaming tokens (AI thoughts), `[PARSE]` (what was extracted), `[SEND]` (display update), `[BUTTON]` (button presses).
