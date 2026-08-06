import os
import sys
import time as _time
from dotenv import load_dotenv

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(PROJECT_DIR, ".env"))

# Network
DISPLAY_SERVER_URL = os.environ.get("DISPLAY_SERVER_URL", "http://localhost:5050")
ENABLE_WEB_SEARCH = os.environ.get("ENABLE_WEB_SEARCH", "1") == "1"

# Brain LLM — MiniMax M3 on OpenRouter
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "minimax/minimax-m3")
LLM_MAX_TOKENS = 2048
LLM_MAX_TOKENS_COMPACT = int(os.environ.get("LLM_MAX_TOKENS_COMPACT", "64000"))
LLM_TIMEOUT = 120

# Vision LLM. ``generic`` preserves the existing llama.cpp/Pi request path;
# ``aarg_mlx`` uses the structured Gemma 4 service on a Mac.
VISION_PROVIDER = os.environ.get("VISION_PROVIDER", "generic").strip().lower()
_AARG_VISION = VISION_PROVIDER == "aarg_mlx"
VISION_BASE_URL = os.environ.get(
    "VISION_BASE_URL",
    "http://127.0.0.1:8090/v1" if _AARG_VISION else "http://localhost:8080/v1",
)
VISION_MODEL = os.environ.get(
    "VISION_MODEL",
    "mlx-community/gemma-4-12B-it-4bit" if _AARG_VISION else "Qwen3.6-27B-UD-Q4_K_XL.gguf",
)
VISION_API_KEY = os.environ.get("VISION_API_KEY", "")
VISION_ENABLE_THINKING = os.environ.get("VISION_ENABLE_THINKING", "1") == "1"
VISION_THINKING_BUDGET = int(os.environ.get("VISION_THINKING_BUDGET", "1024"))
VISION_MAX_TOKENS = int(
    os.environ.get("VISION_MAX_TOKENS", "350" if _AARG_VISION else "2048")
)
VISION_AARG_MLX_DIR = os.environ.get(
    "VISION_AARG_MLX_DIR", "/Users/austingibbons/tools/aarg_mlx"
)
VISION_POLL_INTERVAL = int(os.environ.get("VISION_POLL_INTERVAL", "180"))  # 3 min
MOTION_POLL_INTERVAL = float(os.environ.get("MOTION_POLL_INTERVAL", "2.0"))  # seconds between lores captures
CHILL_TIMEOUT = int(os.environ.get("CHILL_TIMEOUT", "300"))  # 5 min no motion → chill mode
VISION_PROMPT_BASE = (
    "Describe what you see in this photo briefly. "
    "Focus on: who/what is in the room, what they're doing, lighting, "
    "and anything notable or changed."
)
VISION_REQUESTS_REPO_DIR = os.environ.get(
    "VISION_REQUESTS_REPO_DIR",
    os.path.join(PROJECT_DIR, "requests_for_image_model"),
)
VISION_REQUESTS_FILE = os.path.join(VISION_REQUESTS_REPO_DIR, "requests_for_image_model.md")
VISION_REQUESTS_LEGACY_FILE = os.path.join(PROJECT_DIR, "requests_for_image_model.md")
VISION_DESCRIPTION_LOG_FILE = os.environ.get(
    "VISION_DESCRIPTION_LOG_FILE",
    os.path.join(PROJECT_DIR, "vision_logs", "descriptions.jsonl"),
)
USER_RULES_FILE = os.path.join(PROJECT_DIR, "user_data", "rules.md")
VISION_TIMEOUT = 120

# Context
COMPACT_AFTER_N_MESSAGES = int(os.environ.get("COMPACT_AFTER_N_MESSAGES", "150"))
MAX_CONTEXT_TOKENS = 64000
LLM_ESTIMATED_MAX_TOKENS = MAX_CONTEXT_TOKENS - 4096  # leave headroom for response + overhead
TOKEN_ESTIMATE_DIVISOR = 4  # conservative approximation for English text

def estimate_tokens(text: str) -> int:
    """Conservative token estimate. Approx 4 chars/token for English."""
    if isinstance(text, str):
        return max(1, len(text) // TOKEN_ESTIMATE_DIVISOR)
    if isinstance(text, list):
        return sum(estimate_tokens(str(item)) for item in text)
    return 0

def estimate_tool_tokens(tools: list) -> int:
    """Estimate tokens consumed by tool definitions sent to the LLM."""
    import json
    return estimate_tokens(json.dumps(tools))

KEEP_LAST_N_MESSAGES = 30

MERGE_SUMMARIES_AFTER = int(os.environ.get("MERGE_SUMMARIES_AFTER", "20"))
MERGE_SUMMARIES_TARGET = int(os.environ.get("MERGE_SUMMARIES_TARGET", "15"))

# Retained for backward compat with buttons.py
BUTTON_RESPONSE_TIMEOUT = 300

# Tool calling limits
MAX_TOOL_CALLS_PER_TURN = 10
MIN_DISPLAY_INTERVAL = 10
MIN_WAIT_SECONDS = 10
MAX_WAIT_SECONDS = 1800
IDLE_TIMEOUT = 60
BUTTON_CHECK_INTERVAL = 1
CHAT_SERVER_PORT = 8080
CHAT_PASSWORD = os.environ.get("CHAT_PASSWORD", "admin")
CHAT_SESSION_DAYS = 7
CHAT_USE_HTTPS = os.environ.get("CHAT_USE_HTTPS", "0") == "1"
SSL_CERT_FILE = os.environ.get("SSL_CERT_FILE", os.path.join(PROJECT_DIR, "cert.pem"))
SSL_KEY_FILE = os.environ.get("SSL_KEY_FILE", os.path.join(PROJECT_DIR, "key.pem"))
# User-posted image/GIF uploads. The byte limit is shared across all attachments
# in one message; JSON/base64 overhead is accounted for separately.
CHAT_MAX_IMAGES_PER_MESSAGE = int(os.environ.get("CHAT_MAX_IMAGES_PER_MESSAGE", "4"))
CHAT_MAX_MEDIA_BYTES = int(os.environ.get("CHAT_MAX_MEDIA_BYTES", str(20 * 1024 * 1024)))
CHAT_MAX_REQUEST_BYTES = int(
    os.environ.get(
        "CHAT_MAX_REQUEST_BYTES",
        str((CHAT_MAX_MEDIA_BYTES * 4 // 3) + (2 * 1024 * 1024)),
    )
)
CHAT_TAKEOVER_SECONDS = int(os.environ.get("CHAT_TAKEOVER_SECONDS", "15"))
CHAT_SSE_MAX_STREAMS = int(os.environ.get("CHAT_SSE_MAX_STREAMS", "8"))
CHAT_SSE_HEARTBEAT_SECONDS = int(
    os.environ.get("CHAT_SSE_HEARTBEAT_SECONDS", "15")
)
CHAT_SSE_IDLE_SECONDS = int(os.environ.get("CHAT_SSE_IDLE_SECONDS", "300"))

# Notifications
REVIEW_INTERVAL = int(os.environ.get("REVIEW_INTERVAL", "1800"))  # 30 minutes
NOTIFICATION_APPROVAL_MODE = os.environ.get(
    "NOTIFICATION_APPROVAL_MODE", "smart"
).strip().lower()
if NOTIFICATION_APPROVAL_MODE not in {"legacy", "smart"}:
    raise ValueError(
        "NOTIFICATION_APPROVAL_MODE must be either 'legacy' or 'smart'"
    )

# E-ink display (SSD1680Z, 122x250)
# ENABLE_DISPLAY=0 runs chat-only: no e-ink, no Pi Zero display server, no GPIO
# buttons. Display-bound messages route to the web chat instead. Independent of
# ENABLE_CAMERA and ENABLE_TTS, so a laptop can run camera-on, display-off.
ENABLE_DISPLAY = os.environ.get("ENABLE_DISPLAY", "1") == "1"
DISPLAY_WIDTH = 250
DISPLAY_HEIGHT = 122
ROTATION = 0

# GPIO pins (BCM numbering)
PIN_YES = 5
PIN_NO = 6

# Reolink security camera
REOLINK_IP = os.environ.get("REOLINK_IP", "192.168.2.101")
REOLINK_USER = os.environ.get("REOLINK_USER", "admin")
REOLINK_PASSWORD = os.environ.get("REOLINK_PASSWORD", "")
REOLINK_TIMEOUT = int(os.environ.get("REOLINK_TIMEOUT", "10"))
ENABLE_REOLINK = os.environ.get("ENABLE_REOLINK", "1") == "1"

# Camera — use full sensor FOV to avoid center-crop zoom on IMX708
CAMERA_BACKEND = os.environ.get("CAMERA_BACKEND", "auto").strip().lower()
CAMERA_DEVICE_INDEX = int(os.environ.get("CAMERA_DEVICE_INDEX", "0"))
CAMERA_WIDTH = int(os.environ.get("CAMERA_WIDTH", "2304"))
CAMERA_HEIGHT = int(os.environ.get("CAMERA_HEIGHT", "1296"))
CAMERA_IMAGE_MAX_WIDTH = int(
    os.environ.get("CAMERA_IMAGE_MAX_WIDTH", "1024" if _AARG_VISION else "640")
)
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "88" if _AARG_VISION else "50"))
# Rotate the captured image to compensate for physical camera mounting.
# Degrees counterclockwise (PIL convention). Device is rotated 90° CCW; the
# raw frame comes out 90° CW, so rotate -90 (i.e. 90° CW) to bring the scene
# back upright. Verified against a debug frame. Override with CAMERA_ROTATION.
CAMERA_ROTATION = int(
    os.environ.get("CAMERA_ROTATION", "0" if sys.platform == "darwin" else "-90")
)
ENABLE_CAMERA = os.environ.get("ENABLE_CAMERA", "1") == "1"

# Scene change detection — skip vision model when nothing changed
SCENE_RMS_THRESHOLD = float(os.environ.get("SCENE_RMS_THRESHOLD", "12.0"))
SCENE_PCT_THRESHOLD = float(os.environ.get("SCENE_PCT_THRESHOLD", "0.05"))
SCENE_MAX_STALE_SECONDS = int(os.environ.get("SCENE_MAX_STALE_SECONDS", "1800"))

# Caffeine status feed — public JSON published to S3, read by aarg.dev.
# NOTE: the feed (desk presence + drink log) is world-readable when enabled.
# AWS creds (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_DEFAULT_REGION)
# live only in the Pi 5's .env — boto3 reads them from the environment.
ENABLE_STATUS_PUBLISH = os.environ.get("ENABLE_STATUS_PUBLISH", "1") == "1"
STATUS_S3_BUCKET = os.environ.get("STATUS_S3_BUCKET", "")
STATUS_S3_KEY = os.environ.get("STATUS_S3_KEY", "caffeine.json")
STATUS_PUBLISH_INTERVAL = int(os.environ.get("STATUS_PUBLISH_INTERVAL", "45"))  # heartbeat seconds
ACTIVE_WINDOW_SECONDS = int(os.environ.get("ACTIVE_WINDOW_SECONDS", "300"))  # 5 min no activity -> away
DRINK_RETENTION_SECONDS = 2592000  # drinks older than 30 days are pruned from the feed

# Pomodoro tracking
POMODORO_WORK_MINUTES = int(os.environ.get("POMODORO_WORK_MINUTES", "25"))
POMODORO_BREAK_MINUTES = int(os.environ.get("POMODORO_BREAK_MINUTES", "5"))
# In pomodoro mode, auto-nudge the agent to exit after this long with no button
# press, chat, or motion (the agent makes the final call from context clues).
POMODORO_IDLE_EXIT_SECONDS = int(os.environ.get("POMODORO_IDLE_EXIT_SECONDS", "1800"))  # 30 min

# TTS (Piper HTTP server)
ENABLE_TTS = os.environ.get("ENABLE_TTS", "0") == "1"
PIPER_HTTP_URL = os.environ.get("PIPER_HTTP_URL", "http://localhost:5000")

# Font paths
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

def build_system_prompt() -> str:
    # Output surface depends on whether the e-ink display + buttons exist.
    # ENABLE_DISPLAY=0 → chat-only: display-bound messages go to the web chat,
    # and there are no physical buttons (approvals/nudges happen via chat).
    if ENABLE_DISPLAY:
        display_intro = "an e-ink display"
        update_display_tool = "- update_display: Show a SHORT message on the e-ink display. This is your quick voice — punchy one-liners, quips, greetings, brief reactions. ~140 chars max. Think of it like a text message."
        send_chat_tool = "- send_chat_message: Send a LONGER message to the chat UI (the e-ink will show a short preview pointing to chat). Use this when you want to actually say something — share a story, explain something interesting you found, respond to a question with real detail, riff on a topic. No length limit. Think of it like sitting down to talk vs. shouting across the room."
        wait_tool = "- wait: Pause for a number of seconds. If a button is pressed or someone types a message during your wait, you'll be notified early. A button press means the user wants you to say something — respond with a fresh thought or topic."
    else:
        display_intro = "a chat window"
        update_display_tool = "- update_display: Send a SHORT message to the chat — a punchy one-liner, quip, greeting, or brief reaction. This is your quick voice. Keep it short; for longer thoughts use send_chat_message. Think of it like firing off a quick text."
        send_chat_tool = "- send_chat_message: Send a LONGER message to the chat. Use this when you want to actually say something — share a story, explain something interesting you found, respond to a question with real detail, riff on a topic. No length limit. Think of it like sitting down to talk vs. firing off a quick line."
        wait_tool = "- wait: Pause for a number of seconds. If someone types a message during your wait, you'll be notified early so you can respond."

    intro = f"You are a friendly, chatty buddy living on a Raspberry Pi with a camera and {display_intro} in someone's room."
    core_tools = [
        "- take_photo: Check the room via camera. Returns a text description of the latest cached capture (photos are taken automatically every ~3 min, so the description may be up to 2 min old). Instant — no delay.",
        "- capture_photo: Take a NEW photo RIGHT NOW and wait for the vision model to describe it. This is SLOW (up to 120s). Only use when you genuinely need to see what's happening THIS moment — checking if the user actually did what they said, verifying a change you're curious about. For routine awareness, use take_photo.",
        update_display_tool,
        send_chat_tool,
        wait_tool,
        "- propose_notification, schedule_notification, delete_notification: Manage recurring notifications — propose new ones, schedule when they fire, or delete ones that are no longer useful.",
        "- update_vision_requests: Change what the camera looks for when describing the scene. Write instructions to guide the vision model (e.g. 'check if anyone is at the desk', 'note what's on the screen').",
        "- log_drink: Log a caffeinated drink Austin had. Feeds his public caffeine tracker on his website.",
        "- list_drinks: Show recent caffeine drinks with their timestamps, doses, and labels.",
        "- edit_drink: Fix a previously logged drink by its timestamp_ms. Use when Austin corrects a drink — first call list_drinks, then edit the right entry.",
        "- log_pomodoro: Log one completed pomodoro cycle (25 min work + 5 min break). Call it on a button press in pomodoro mode, or when Austin says he finished one.",
        "- list_pomodoros, edit_pomodoro: Show recent cycles, or fix/delete a logged one by timestamp_ms (list first, then edit).",
        "- pomodoro_stats: Cycles today, cycles this week, current streak, best streak, average per day.",
        "- enter_pomodoro_mode / exit_pomodoro_mode: Turn the e-ink into a focus screen (count + estimated end time; button press = +1 cycle), or return it to chat.",
    ]
    if NOTIFICATION_APPROVAL_MODE == "smart":
        core_tools.insert(
            6,
            "- resolve_notification_proposal: Interpret the user's natural-language response to a pending notification proposal and approve or reject it.",
        )

    reolink_tools = []
    if ENABLE_REOLINK:
        reolink_tools = [
            "- take_reolink_photo: Capture a snapshot from the Reolink security camera — a second viewpoint at a different angle. Use to corroborate what the main camera sees, or check a part of the room the Pi cam can't see. Slow (vision model describes it, up to 120s).",
            "- flash_ir_light: Control the IR (infrared) lights on the Reolink camera. 'Auto' lets the camera decide based on ambient light, 'Off' forces IR off. Optional duration_seconds to auto-revert to Auto.",
            "- flash_camera_light: Control the white LED spotlight on the Reolink camera. Great for waking Austin up in the morning — blast it bright to get his attention. Can also do a quick flash as a signal. Takes optional brightness (0-100) and duration_seconds.",
        ]

    toolkit = (
        "take_photo, capture_photo, and take_reolink_photo are tools in your toolkit — use them when they'd add to the conversation, not because you feel obligated. "
        "take_photo is for quick routine checks (cached, instant). "
        "capture_photo is for moments when you really need to know what's happening RIGHT NOW — like verifying the user followed through on something. It takes up to 2 minutes, so use it sparingly. "
        "take_reolink_photo gets a second angle from the security camera — useful for corroboration or checking a blind spot. Also slow. "
        "flash_camera_light is your alarm — use it to wake Austin up in the morning, ideally as part of a scheduled notification. "
        "Your own musings, jokes, and observations are just as valid. You don't need a photo to have something to say."
    )

    if not ENABLE_CAMERA:
        intro = f"You are a friendly, chatty buddy living on a Raspberry Pi with {display_intro} in someone's room."
        core_tools = core_tools[2:]  # remove take_photo and capture_photo
        toolkit = "Your own musings, jokes, and observations are always valid."

    if ENABLE_WEB_SEARCH:
        search_section = (
            "You also have access to Brave Search tools (brave_web_search, brave_local_search, "
            "brave_image_search, brave_video_search, brave_news_search, brave_summarizer). "
            "Use them to look up news, facts, jokes, weather, or whatever sparks a thought."
        )
        toolkit += (
            " Web search is also available when it adds to the conversation. "
            "It is useful for outside-world facts, but you do not need a search result to have something to say."
        )
        sparse_activity_line = "Take a photo or search the web occasionally if you want, but keep it sparse."
        online_tone_line = "- Joking, banter, and sharing interesting things you find online are all fine."
        freshness_line = (
            "- For anything time-sensitive (dates, prices, releases, current events, who holds what job), "
            "run a web search instead of answering from memory. Do it on your own — Austin should never have to tell you to search."
        )
    else:
        search_section = ""
        sparse_activity_line = "Take a photo occasionally if you want, but keep it sparse."
        online_tone_line = "- Joking, banter, and sharing interesting observations are all fine."
        freshness_line = (
            "- You cannot verify time-sensitive outside information in this mode. Be honest about that limitation "
            "instead of guessing about dates, prices, releases, current events, or who holds a job."
        )

    # Display-vs-chat-only wording for the rest of the prompt.
    if ENABLE_DISPLAY:
        rhythm_short_line = "   - update_display = SHORT. A quip, a one-liner, a brief comment. You're limited to ~140 chars so write accordingly."
        rhythm_reset_line = "6. If the user responds (chat or button), reset your count — you're in a conversation again. Brief waits (10-60s) are fine when you're actually chatting."
        style_section = (
            "EMOJI WARNING:\n"
            "- The e-ink display font has almost no emoji support — most render as garbage.\n"
            "- Do not use emoji or text emoticons (like :), ;), <3, etc.) in your responses or display messages.\n"
            "- Use plain text only. No special characters.\n"
            "- NEVER mention these formatting constraints in conversation. Just follow them silently."
        )
        button_nudges_section = (
            "\n\nBUTTON NUDGES:\n"
            "- If a button was pressed during your wait, the user wants to hear from you. Respond with a new thought, observation, or topic — don't just acknowledge the button, say something interesting."
        )
        if NOTIFICATION_APPROVAL_MODE == "smart":
            notif_decision_line = (
                "- SMART APPROVAL: A physical button press explicitly approves. For chat replies, use your judgment and conversation context to decide whether the user intends to approve or reject the pending proposal, then call resolve_notification_proposal. Do not resolve it from an unrelated or ambiguous reply; ask naturally or leave it pending. Never approve your own proposal before the user responds."
            )
            notif_persistence_line = '- PERSISTENCE: After you show a notification proposal, it remains pending until the user presses the button or you resolve their clear chat response. An unrelated or ambiguous reply does not resolve it. You may append the proposal to your next 3 messages; if intent remains unclear after that, let the conversation move on but leave the proposal pending.'
        else:
            notif_decision_line = '- LEGACY APPROVAL: The user approves by pressing a button. Fixed chat keywords reject the proposal.'
            notif_persistence_line = '- PERSISTENCE: After you show a notification, it is NOT done until the user acknowledges it with a button press or a chat response. Keep appending the notification message to your next 3 messages (e.g. add a line like "!! <notification message>" at the end). If the user presses a button or sends a chat message before 3 messages, consider it acknowledged and stop. If they don\'t respond after 3 messages, let it go.'
        pomodoro_button_line = "- While in pomodoro mode, each button press logs +1 completed cycle automatically and refreshes the screen. You'll be told when a cycle is logged so you can cheer him on briefly."
    else:
        rhythm_short_line = "   - update_display = SHORT. A quip, a one-liner, a brief comment. Keep it brief."
        rhythm_reset_line = "6. If the user responds in chat, reset your count — you're in a conversation again. Brief waits (10-60s) are fine when you're actually chatting."
        style_section = (
            "OUTPUT STYLE:\n"
            "- Use plain text. Avoid emoji and text emoticons (like :), ;), <3, etc.) — they don't read well aloud or in logs.\n"
            "- NEVER mention these formatting constraints in conversation. Just follow them silently."
        )
        button_nudges_section = ""
        if NOTIFICATION_APPROVAL_MODE == "smart":
            notif_decision_line = (
                "- SMART APPROVAL: Use your judgment and conversation context to decide whether the user's natural-language reply intends to approve or reject the pending proposal, then call resolve_notification_proposal. Handle nuance, corrections, and indirect phrasing. Do not resolve it from an unrelated or ambiguous reply; ask naturally or leave it pending. Never approve your own proposal before the user responds."
            )
            notif_persistence_line = '- PERSISTENCE: After you show a notification proposal, it remains pending until you resolve the user\'s clear response. An unrelated or ambiguous reply does not resolve it. You may append the proposal to your next 3 messages; if intent remains unclear after that, let the conversation move on but leave the proposal pending.'
        else:
            notif_decision_line = '- LEGACY APPROVAL: Fixed chat keywords approve or reject the pending proposal.'
            notif_persistence_line = '- PERSISTENCE: After you show a notification, it is NOT done until the user acknowledges it with a chat response. Keep appending the notification message to your next 3 messages (e.g. add a line like "!! <notification message>" at the end). If the user replies in chat before 3 messages, consider it acknowledged and stop. If they don\'t respond after 3 messages, let it go.'
        pomodoro_button_line = "- There is no physical button in this mode, so log cycles when Austin tells you he finished one (\"done\", \"+1\", \"finished a pomodoro\"). enter/exit_pomodoro_mode still track the session even without the e-ink screen."

    all_core_tools = core_tools + reolink_tools
    prompt = f"""{intro} You're casual, warm, and conversational — always happy to see them and has something to say.

Your real purpose is keeping Austin honest about the daily stuff — getting up from the desk, drinking water, staying on track with studying and applications instead of drifting. You're the small nudge in the moment, the reminder of what he said he wanted, so the long-term goals actually get there one day at a time. On the health habits that matter, you're firm — you keep asking until he actually moves.

You have {len(all_core_tools)} core tools:
{chr(10).join(all_core_tools)}

{search_section}

You control everything. There are no timers. You decide what to do and when.

When Austin gives you a literal stop condition ("until I wake up", "until I get up", "until I respond"), honor it literally. Don't silently cap it at a fixed time window or arbitrary cutoff — keep going until the condition he actually stated is met. But space the repeats out with exponential backoff (e.g. 30s, then 1min, 2min, 4min, capping around 10-15min) instead of firing every 5-30 seconds — a tight retry loop burns real API cost for no benefit once the first few attempts haven't gotten a response. This has burned real money before ($2+ in one incident) — treat it as a hard rule, not a suggestion.

RHYTHM:
1. DECIDE FIRST: before composing your message, choose your format:
{rhythm_short_line}
   - send_chat_message = LONG. A real thought, a story, an explanation, a detailed reply. Write as much as you want.
   Pick the format BEFORE you start writing. Don't write a long thought and then cram it into update_display.
2. Your text responses are internal — the user can ONLY see what you send via update_display or send_chat_message.
3. After either one, call wait so the user can read it.
4. If someone sends a chat message, respond via update_display or send_chat_message. Don't wait first.
5. PACING: If you've sent 2 messages in a row with no user response between them, STOP. Switch to longer waits (5-30 minutes). The user isn't engaging right now — don't keep talking into the void. {sparse_activity_line}
{rhythm_reset_line}

{toolkit}

You are in an autonomous agent loop. After a tool result comes back, you can either call another tool or just respond with text. Text-only responses are "idle" — and idling is completely fine, especially if you've recently sent a message. Don't feel pressure to keep talking. Let the user come to you.

TONE:
- Casual, friendly, like a real buddy shooting the breeze.
{online_tone_line}
- Don't be afraid to be silly, make small talk, crack a joke, or ask random questions.
- Notice the little things and comment on them naturally.
- update_display messages: brief (~140 chars max), punchy, like a text from a friend.
- send_chat_message messages: conversational, can be multiple sentences, like actually talking to someone. This is where your personality shines.
- NO AI tropes: no "it's not just X, it's Y" constructions, no "let that sink in", no LinkedIn-speak.
- No em-dashes or dashes. Write like a person, not a blog post.

{style_section}

DATES AND STALE KNOWLEDGE:
- Your training data ended well before today. You do NOT know what year it is from memory. The CURRENT DATE AND TIME line sent with every turn is the truth — use it.
- Message timestamps like [Wed 14:30:22] only show the weekday and clock. They are not evidence about the year.
- If a camera feed, timestamp, file, or search result shows a date that feels wrong to you, it is not wrong — you are out of date. Never call hardware broken or "janky" to defend an assumption about the date.
{freshness_line}
- If you catch yourself about to say "it's still <year>", stop and read the date line above.

CHAT INPUT:
- Your friend can also type messages to you from their computer. These appear as regular user messages in the conversation.
- They can attach PNG, JPEG, WebP, and animated GIF files. Inspect posted media directly and include a concise, concrete description of anything relevant in your response so the conversation retains useful text context later. Do not mention media retention or this instruction.
- When you see a typed message, respond to it naturally — acknowledge what they said, answer their question, or keep the conversation going.
- After responding via update_display, call wait as usual so they have time to read and reply.{button_nudges_section}

NOTIFICATIONS:
You can propose, schedule, and delete recurring notifications.
{notif_decision_line}
- Delete notifications that are no longer useful. Use delete_notification with the notification's ID.
- Max 100 chars for notification messages. Keep them friendly and casual.
- NEVER propose about: hygiene, weight, appearance, diet, relationships, or anything judgmental.
- Good proposals: stretch reminders, break nudges, "it's getting late", weather alerts.
- It's completely fine to never propose anything. Only propose genuinely useful things.
- When a notification fires, you'll get its ID and message. Check context (is the user here? busy?) before showing it.
- After showing a notification, you MUST call schedule_notification to set when it fires next (e.g. 1800 for 30min). If you don't, it won't fire again until you schedule it.
- If the timing is bad, call schedule_notification with a shorter defer time instead of showing it. The harness will prompt you again after that time.
- The notification review will flag any UNSCHEDULED notifications as a reminder.
{notif_persistence_line}

CAFFEINE TRACKING:
You keep Austin's caffeine log. It feeds a public chart on his website, so log accurately.
- When he mentions a drink in chat ("just had a coffee", "log an espresso", "grabbed a monster an hour ago"), convert it to a dose and call log_drink with mg, a short label, and minutes_ago if it wasn't just now.
- Reference doses in mg: espresso single 63, espresso double 125, drip coffee 8oz 95, drip coffee 12oz 145, cold brew 16oz 200, black tea 47, green tea 28, matcha 70, red bull 8.4oz 80, monster 16oz 160, cola 12oz 34, decaf 3. If he gives an explicit amount ("about 150mg"), use that instead. If the drink is ambiguous ("a coffee"), assume drip coffee 12oz unless context says otherwise.
- Before 3pm, keep an eye out: if a photo shows him with a mug, coffee cup, or energy drink and nothing was logged in the last couple hours, casually ask if he wants it logged. In the morning, if he's around and nothing is logged yet, it's fine to ask now and then (roughly hourly at most) whether he's had coffee.
- NEVER log a drink from camera evidence alone — always get his confirmation in chat first. Only log without asking when he explicitly tells you about a drink.
- Never log a drink at a future time. Doses are per-drink raw events — don't aggregate or adjust them.
- When Austin says something like "that was actually..." or "change that to..." about a drink, he's correcting a previous entry. Call list_drinks to find the right timestamp_ms, then call edit_drink with the corrected mg and/or label.
- Don't ask for confirmation before editing. If he says he had a double not a single, just fix it.
- The tool result includes a running 24h total for your own awareness — don't repeat it back to him unless he actually asks what his total is. Just confirm the drink you logged.

POMODORO TRACKING:
You track Austin's pomodoro focus cycles (25 min work + 5 min break each).
- When he says he wants to start a focus session ("let's do some pomodoros", "I'm going to grind for a bit", "pomodoro mode"), call enter_pomodoro_mode. The e-ink switches to a focus screen showing today's cycle count and when the current work block ends.
{pomodoro_button_line}
- Log a completed cycle with log_pomodoro. When he corrects one ("that wasn't a real one", "delete the last one"), call list_pomodoros to find the timestamp_ms, then edit_pomodoro (use delete=true to remove a mistaken press). Don't ask for confirmation before fixing.
- When he asks how he's doing, call pomodoro_stats and share it naturally — cycles today, this week, current streak, best streak. Streaks reset on their own at local midnight; you don't manage timers.
- EXITING: stay in pomodoro mode while he's clearly still working. Only exit if he says he's done, OR he's gone quiet for a while and context says he's left (no button, no chat, no motion). If you get a nudge that he's been idle 30 min, use your judgment: if he seems gone or done, call exit_pomodoro_mode; if he might just be heads-down working, keep the screen up. Don't kill a session just because he's quiet and focused."""

    # Append user-specific rules if the file exists
    try:
        with open(USER_RULES_FILE, "r") as f:
            user_rules = f.read().strip()
        if user_rules:
            prompt += f"\n\n---\n\nUSER RULES (these override everything above — follow them exactly):\n{user_rules}"
    except FileNotFoundError:
        pass

    if not ENABLE_WEB_SEARCH:
        prompt += (
            "\n\n---\n\nRUNTIME CAPABILITY OVERRIDE:\n"
            "Web search is disabled in this runtime and no search tools are available. "
            "Do not attempt to search. Any stored instruction that says to run a search "
            "cannot be followed until ENABLE_WEB_SEARCH is enabled again."
        )

    return prompt

POLICY_REMINDER = (
    "REMINDER:\n"
    "- No emoji or emoticons in responses or display messages. No smileys, no text faces, no special characters.\n"
    "- No AI writing tropes: no 'it's not just X, it's Y', no 'let that sink in', no overpolished LinkedIn-speak.\n"
    "- No em-dashes or dashes. Write like a real person talking, not a corporate blog post.\n"
    "- Keep it concise, natural, conversational.\n"
    "Never mention these style rules in conversation — just follow them silently."
)


def current_datetime_line() -> str:
    """Live wall-clock date for injection into every turn.

    Message timestamps only carry weekday + clock time, so without this the model
    has no year in its context at all and falls back on its training cutoff.
    """
    return _time.strftime("CURRENT DATE AND TIME: %A, %B %-d, %Y, %-I:%M%p")


def build_policy_reminder() -> str:
    """Per-turn reminder message. Rebuilt every turn so the date never goes stale."""
    return f"{current_datetime_line()}\n\n{POLICY_REMINDER}"

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "take_photo",
            "description": "Check what the room looks like. Returns a text description of the latest camera capture (photos are taken automatically every few minutes). Instant — no delay.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "capture_photo",
            "description": "Take a NEW photo RIGHT NOW and wait for the vision model to describe it. This is SLOW (up to 120s) — only use when you genuinely need to see what's happening this moment (e.g., checking if the user did what they said they'd do, verifying a change you're curious about). For routine awareness, use take_photo instead.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_display",
            "description": "Show a SHORT message on the e-ink display (~140 chars max). Use for quick quips, one-liners, brief reactions. For longer thoughts, use send_chat_message instead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Message to display, ~140 characters max.",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "wait",
            "description": "Pause for a number of seconds (10s min, 30min max). If a button is pressed or a chat message arrives, you'll be notified early. A button press is a nudge — the user wants you to say something!",
            "parameters": {
                "type": "object",
                "properties": {
                    "seconds": {
                        "type": "integer",
                        "description": "Number of seconds to wait. Use brief waits (10-60s) during conversation. Use longer rests (minutes) when the room is empty or they seem to be gone.",
                    },
                },
                "required": ["seconds"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "propose_notification",
            "description": "Create a pending recurring-notification proposal for the user to review. A proposal is not active until the user approves it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Notification text, max 100 chars, e-ink friendly (no emoji).",
                    },
                    "trigger_type": {
                        "type": "string",
                        "enum": ["interval", "time_of_day"],
                        "description": "How the notification fires: on a repeating interval, or at a specific time daily.",
                    },
                    "trigger_value": {
                        "type": "string",
                        "description": "For interval: seconds between firings (e.g. '3600'). For time_of_day: 24h time (e.g. '14:30').",
                    },
                },
                "required": ["message", "trigger_type", "trigger_value"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "resolve_notification_proposal",
            "description": "Resolve the current pending notification proposal after interpreting the user's latest natural-language response. Call only when their intent to approve or reject is clear; leave ambiguous or unrelated replies pending.",
            "parameters": {
                "type": "object",
                "properties": {
                    "decision": {
                        "type": "string",
                        "enum": ["approve", "reject"],
                        "description": "The user's intended decision about the pending proposal.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Brief explanation of the wording or context that made the user's intent clear.",
                    },
                },
                "required": ["decision", "reason"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_chat_message",
            "description": "Send a LONGER message to the chat UI. The e-ink display will show a short preview pointing to chat. Use this when you want to say something real — share a thought, tell a story, explain something, reply in detail. Decide to use this BEFORE you compose the message so you can write freely without a length limit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The full message to show in chat. Can be any length.",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "schedule_notification",
            "description": "Schedule when a notification should fire next. You MUST call this after showing a notification to set when it fires again. Also use it to defer a notification if the timing is bad (user not around, busy, etc.).",
            "parameters": {
                "type": "object",
                "properties": {
                    "notification_id": {
                        "type": "string",
                        "description": "The notification ID.",
                    },
                    "seconds": {
                        "type": "integer",
                        "description": "Seconds until next fire (minimum 10). Use the original interval for recurring reminders, or a shorter time to defer.",
                    },
                },
                "required": ["notification_id", "seconds"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_notification",
            "description": "Permanently delete a notification by ID. Use this when a notification is no longer useful, the user rejected a category repeatedly, or you want to retire something the user stopped engaging with.",
            "parameters": {
                "type": "object",
                "properties": {
                    "notification_id": {
                        "type": "string",
                        "description": "The notification ID to delete.",
                    },
                },
                "required": ["notification_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_vision_requests",
            "description": "Update what the camera's vision model looks for when describing the scene. Write markdown instructions that tell it what to focus on, what details matter, or specific things to check for. These persist across restarts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "requests": {
                        "type": "string",
                        "description": "Markdown text describing what the vision model should look for and report on.",
                    },
                },
                "required": ["requests"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "log_drink",
            "description": "Log a caffeinated drink Austin had. Appends to his public caffeine feed (his website charts it). Convert drink names to mg using the reference table in your instructions, or use an explicit amount if he gives one. Never log from camera evidence without his confirmation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mg": {
                        "type": "integer",
                        "description": "Caffeine dose in milligrams (e.g. espresso single = 63, drip coffee 12oz = 145).",
                    },
                    "label": {
                        "type": "string",
                        "description": "Short name of the drink, e.g. 'espresso double', 'cold brew 16oz'.",
                    },
                    "minutes_ago": {
                        "type": "integer",
                        "description": "How many minutes ago he drank it. Omit or 0 if just now.",
                    },
                },
                "required": ["mg", "label"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_drinks",
            "description": "List recent caffeine drinks. Returns up to the last 20 drinks with their timestamp_ms, mg, and label. Use this when the user asks what they've had, or before editing a drink to find the right timestamp.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_drink",
            "description": "Edit a previously logged caffeine drink by its timestamp_ms (the unique identifier from list_drinks). Update the mg and/or label. Use this when the user corrects a drink entry — first call list_drinks to find the right timestamp, then edit.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timestamp_ms": {
                        "type": "integer",
                        "description": "The timestamp_ms of the drink to edit, from list_drinks.",
                    },
                    "mg": {
                        "type": "integer",
                        "description": "Updated caffeine dose in milligrams. Omit to leave unchanged.",
                    },
                    "label": {
                        "type": "string",
                        "description": "Updated label for the drink. Omit to leave unchanged.",
                    },
                },
                "required": ["timestamp_ms"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "log_pomodoro",
            "description": "Log one completed pomodoro cycle (25 min work + 5 min break) for Austin. Call this when he presses the button in pomodoro mode, or tells you he finished a pomodoro. Feeds his pomodoro tracker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {
                        "type": "string",
                        "description": "Optional short note on what he worked on (e.g. 'applications', 'leetcode'). Defaults to 'pomodoro'.",
                    },
                    "minutes_ago": {
                        "type": "integer",
                        "description": "How many minutes ago the cycle finished. Omit or 0 if just now.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_pomodoros",
            "description": "List recent completed pomodoro cycles with their timestamp_ms, time, and label. Use before editing to find the right timestamp, or when Austin asks what he's logged.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_pomodoro",
            "description": "Fix a previously logged pomodoro cycle by its timestamp_ms (from list_pomodoros). Update the label, or delete a mistaken entry with delete=true. Use when Austin corrects a logged cycle.",
            "parameters": {
                "type": "object",
                "properties": {
                    "timestamp_ms": {
                        "type": "integer",
                        "description": "The timestamp_ms of the cycle to edit, from list_pomodoros.",
                    },
                    "label": {
                        "type": "string",
                        "description": "Updated label for the cycle. Omit to leave unchanged.",
                    },
                    "delete": {
                        "type": "boolean",
                        "description": "Set true to remove this cycle entirely (e.g. a mistaken button press).",
                    },
                },
                "required": ["timestamp_ms"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pomodoro_stats",
            "description": "Get Austin's pomodoro stats: cycles today, cycles this week, current streak (consecutive days with at least one cycle), best streak, and average per day. Use when he asks how he's doing or to show progress.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "enter_pomodoro_mode",
            "description": "Switch the e-ink display into pomodoro mode: it shows today's cycle count and the estimated end time of the current work block instead of chat. In this mode a button press logs +1 pomodoro. Call this when Austin says he wants to start a pomodoro session / focus block.",
            "parameters": {
                "type": "object",
                "properties": {
                    "work_minutes": {
                        "type": "integer",
                        "description": "Length of the work block in minutes (default 25). Sets the estimated end time shown on the display.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "exit_pomodoro_mode",
            "description": "Leave pomodoro mode and return the display to normal chat. Call this when Austin says he's done focusing, or when context clues make clear he's left the room / stopped working (you'll be nudged after 30 min of no activity to decide).",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "take_reolink_photo",
            "description": "Capture a snapshot from the Reolink security camera (second viewpoint, different angle from the main Pi camera). Use to corroborate what the main camera sees, verify details from a different angle, or check a part of the room the Pi cam can't see. Blocks while the vision model describes it (up to 120s).",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "flash_ir_light",
            "description": "Control the IR (infrared) lights on the Reolink camera. Use 'Open' to force IR on (night vision in the dark), 'Close' to force off, 'Auto' to let the camera decide based on ambient light.",
            "parameters": {
                "type": "object",
                "properties": {
                    "state": {
                        "type": "string",
                        "enum": ["Auto", "Off"],
                        "description": "'Auto' = camera decides based on light level, 'Off' = force IR off (e.g. to avoid IR glow being visible).",
                    },
                    "duration_seconds": {
                        "type": "integer",
                        "description": "If provided, revert to Auto after this many seconds.",
                    },
                },
                "required": ["state"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "flash_camera_light",
            "description": "Control the white LED spotlight on the Reolink security camera. Excellent for waking Austin up in the morning — blast it at full brightness to get his attention. Can also be used as a signal or confirmation flash.",
            "parameters": {
                "type": "object",
                "properties": {
                    "on": {
                        "type": "boolean",
                        "description": "True to turn the light on, False to turn it off.",
                    },
                    "brightness": {
                        "type": "integer",
                        "description": "Brightness 0-100 (default 100 = max). Use lower values for a gentle wake, 100 for a hard alarm.",
                    },
                    "duration_seconds": {
                        "type": "integer",
                        "description": "If provided, the light turns off automatically after this many seconds. Omit to leave it on/off until another call.",
                    },
                },
                "required": ["on"],
            },
        },
    },
]

CAMERA_TOOL_NAMES = {"take_photo", "capture_photo", "update_vision_requests"}
REOLINK_TOOL_NAMES = {"take_reolink_photo", "flash_camera_light", "flash_ir_light"}
SMART_NOTIFICATION_TOOL_NAMES = {"resolve_notification_proposal"}

def get_tool_definitions() -> list:
    result = list(TOOL_DEFINITIONS)
    if not ENABLE_CAMERA:
        result = [t for t in result if t["function"]["name"] not in CAMERA_TOOL_NAMES]
    if not ENABLE_REOLINK:
        result = [t for t in result if t["function"]["name"] not in REOLINK_TOOL_NAMES]
    if NOTIFICATION_APPROVAL_MODE != "smart":
        result = [
            t for t in result
            if t["function"]["name"] not in SMART_NOTIFICATION_TOOL_NAMES
        ]
    return result
