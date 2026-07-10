# Feature Ideas — Taking the Project to the Next Level

*Written 2026-07-10 by Claude after a full read-through of the codebase.*

I've read through the whole system — the orchestrator loop, the two-model split, the vision/motion tiering, notifications, caffeine feed, context compaction, the chat server, and `harvest_brain.py`. Here's my honest evaluation and where I'd take it.

## Quick evaluation

This is a genuinely well-designed system for what it is. The strongest ideas: the two-tier vision loop (cheap lores motion detection gating expensive vision calls, with chill/active modes), the cached `take_photo` vs blocking `capture_photo` split, `_repair_pairing()` for OpenRouter's strict format, and the notification propose/approve/schedule protocol where the *model* owns rescheduling. `harvest_brain.py` shows you're already thinking about the right next thing.

The core weakness: **the agent has no durable knowledge**. Everything it learns about Austin lives in `context.json`, which gets compacted, then summary-merged, then compacted again — a lossy funnel that guarantees the buddy has amnesia on a ~2-week horizon. Its stated purpose (habits, studying, applications) is long-horizon, but its memory is short-horizon. Most of my suggestions attack that mismatch from different angles.

## The 10 features

**1. Durable long-term memory (`remember` / `recall` tools + a daily journal).** The single highest-impact change. Give the agent a `remember(fact, category)` tool writing to `user_data/memory.md` (or per-category files), inject the memory file into the system prompt the way `rules.md` already is, and add a nightly step that distills the day's context into a dated journal entry *before* compaction destroys it. A roommate who remembers "you said you'd finish the application by Friday" a week later is a categorically different product than one who forgets by Thursday. The `USER_RULES_FILE` mechanism in `config.py:244` is already the template for how to wire it in.

**2. Generalize the caffeine log into a habit ledger.** `DrinkStore` is a great pattern — append-only events, retention, public feed — but caffeine is the only *structured* thing the agent tracks, while its actual mission is water, breaks, study, and applications. A generic `log_event(type, value, label)` store (water, break_taken, study_session, application_sent, woke_up) gives the agent facts to nudge with: "you logged 6 study hours Monday, zero since" hits much harder than a vague reminder. It also feeds aarg.dev with more than caffeine.

**3. A structured presence timeline from the vision loop.** The motion loop already runs every 2s and the vision model already describes scenes, but nothing aggregates it. Have each scene description pass through a tiny extraction step producing `{at_desk, standing, screen_on, ...}` appended to a timeline file. Then "how long has Austin been sitting?" becomes a queryable fact instead of the heuristic in `_detect_patterns()` (main.py:902), and the "get up from the desk" nudge — the agent's *primary job* — can fire on real continuous-sitting data. This is the difference between a buddy that guesses and one that knows.

**4. Voice input.** You already have the output half (Piper TTS in `tts.py`); the input half is missing. Whisper (faster-whisper on the Pi 5, or whisper.cpp on the llama server) with either a wake word or push-to-talk on one of the GPIO buttons turns typing-at-a-webpage into actually talking to your roommate. This is the biggest *interaction* upgrade available, and the plumbing (chat_queue, chat_event) already treats input as text, so transcription slots in cleanly.

**5. Harness-driven daily rhythm: morning briefing and evening review.** Right now time-of-day awareness depends on the model noticing timestamps. Inject two scheduled events from the harness (like the notification review interrupt in `_tool_wait`): a morning event ("it's 8am — greet him, weather, what's the plan today, anything carried over from yesterday's journal") and an evening one ("recap the day's habit ledger, ask how it went"). Combined with #1 and #2, this creates the accountability loop the system prompt describes but can't currently deliver. The evening review is also the natural trigger for the journal distillation.

**6. Calendar integration.** You already have the MCP client infrastructure for Brave Search — add a Google Calendar MCP (or a small CalDAV/ICS poller). "Your interview is in 40 minutes and you're still in pajamas" is the killer app of a camera-equipped desk buddy, and it makes wake-up alarms (the Reolink spotlight) schedule themselves against real commitments instead of guesses.

**7. Close the self-improvement loop.** `harvest_brain.py` is excellent but manual. Run it nightly via cron over the pre-compaction context, and have it emit *proposed amendments* to `user_data/rules.md` that Austin approves in chat (or even a button press). You built the analysis half; the missing half is the write-back path. An agent that measurably gets less annoying week over week is a rare thing.

**8. A real dashboard tab in the chat UI.** The chat server is the natural home for: the latest debug photo (they're already saved to `debug_images/`), what the e-ink currently shows, chill/active mode, context token gauge, active notifications with next-fire times, and habit/caffeine charts. Right now the only window into the agent's head is `journalctl`. This costs little (one more route serving JSON + a static page) and pays off every time something feels "off" — including the known e-ink stuck-display issue, which you'd *see* immediately because the dashboard mirror and the physical display would disagree.

**9. Reliability hardening — three specific ones.** (a) `Context.save()` writes `context.json` non-atomically; a crash mid-write destroys the agent's entire memory. Write to a temp file + `os.rename`, and keep a rotating `.bak`. (b) The vision thread has no watchdog — if `capture_lores()` starts failing permanently, the agent silently goes blind; detect a stale `latest_scene` and tell the agent about it. (c) A display watchdog: the display server returns 200 OK even when the panel is stuck (your known hardware issue) — add a refresh counter/heartbeat check so the agent knows its words aren't reaching the screen and can fall back to chat/TTS.

**10. Study-session mode (pomodoro with camera verification).** Directly on-mission: a `start_focus_session(minutes, goal)` tool that logs to the habit ledger, suppresses all chatter and notifications for the duration (a real "do not disturb" state the harness enforces, not just prompt guidance), then checks the camera at the end — "did he actually study or was he on his phone?" — and logs the outcome. It gives the agent a way to be *useful during* focused work rather than only between it, and the accountability check-in at the end is exactly the "firm on habits" behavior the system prompt asks for.

## If I had to pick three

Memory (#1), the habit ledger (#2), and the presence timeline (#3). They're mutually reinforcing: the timeline generates facts, the ledger structures them, and memory persists them — together they transform the agent from "chatty camera that forgets" into the accountability partner the system prompt already describes. Voice (#4) is the biggest *feel* upgrade if you want something more fun first.
