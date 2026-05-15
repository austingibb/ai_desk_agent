import os
from dotenv import load_dotenv

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(PROJECT_DIR, ".env"))

# Network
DISPLAY_SERVER_URL = os.environ.get("DISPLAY_SERVER_URL", "http://localhost:5050")

# Brain LLM — DeepSeek on OpenRouter
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek/deepseek-chat")
LLM_MAX_TOKENS = 2048
LLM_MAX_TOKENS_COMPACT = 1024
LLM_TIMEOUT = 120

# Vision LLM — local Gemma on llama.cpp
VISION_BASE_URL = os.environ.get("VISION_BASE_URL", "http://localhost:8081/v1")
VISION_MODEL = os.environ.get("VISION_MODEL", "gemma-4-31B-it-UD-Q4_K_XL.gguf")
VISION_API_KEY = os.environ.get("VISION_API_KEY", "")
VISION_POLL_INTERVAL = int(os.environ.get("VISION_POLL_INTERVAL", "180"))  # 3 min
VISION_PROMPT_BASE = (
    "Describe what you see in this photo briefly. "
    "Focus on: who/what is in the room, what they're doing, lighting, "
    "and anything notable or changed."
)
VISION_REQUESTS_FILE = os.path.join(PROJECT_DIR, "requests_for_image_model.md")
VISION_TIMEOUT = 120

# Context
COMPACT_AFTER_N_MESSAGES = int(os.environ.get("COMPACT_AFTER_N_MESSAGES", "150"))
MAX_CONTEXT_TOKENS = 64000
LLM_ESTIMATED_MAX_TOKENS = MAX_CONTEXT_TOKENS - 4096  # leave headroom for response + overhead
TOKEN_ESTIMATE_DIVISOR = 4  # ~4 chars per token for DeepSeek

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

# Notifications
REVIEW_INTERVAL = int(os.environ.get("REVIEW_INTERVAL", "1800"))  # 30 minutes
MAX_PROPOSAL_INTERVAL = 7200  # 2 hours — min time between proposals
CATEGORY_COOLDOWN_REVIEWS = 3  # after proposing in a category, skip it for N reviews

# E-ink display (SSD1680Z, 122x250)
DISPLAY_WIDTH = 250
DISPLAY_HEIGHT = 122
ROTATION = 0

# GPIO pins (BCM numbering)
PIN_YES = 5
PIN_NO = 6

# Camera — use full sensor FOV to avoid center-crop zoom on IMX708
CAMERA_WIDTH = 2304
CAMERA_HEIGHT = 1296
JPEG_QUALITY = 50
ENABLE_CAMERA = os.environ.get("ENABLE_CAMERA", "1") == "1"

# Font paths
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

def build_system_prompt() -> str:
    intro = "You are a friendly, chatty buddy living on a Raspberry Pi with a camera and an e-ink display in someone's room."
    core_tools = [
        "- take_photo: Check the room via camera. Returns a text description of the latest cached capture (photos are taken automatically every ~3 min, so the description may be up to 2 min old). Instant — no delay.",
        "- capture_photo: Take a NEW photo RIGHT NOW and wait for the vision model to describe it. This is SLOW (up to 120s). Only use when you genuinely need to see what's happening THIS moment — checking if the user actually did what they said, verifying a change you're curious about. For routine awareness, use take_photo.",
        "- update_display: Show a SHORT message on the e-ink display. This is your quick voice — punchy one-liners, quips, greetings, brief reactions. ~140 chars max. Think of it like a text message.",
        "- send_chat_message: Send a LONGER message to the chat UI (the e-ink will show a short preview pointing to chat). Use this when you want to actually say something — share a story, explain something interesting you found, respond to a question with real detail, riff on a topic. No length limit. Think of it like sitting down to talk vs. shouting across the room.",
        "- wait: Pause for a number of seconds. If a button is pressed or someone types a message during your wait, you'll be notified early. A button press means the user wants you to say something — respond with a fresh thought or topic.",
        "- propose_notification, schedule_notification, delete_notification: Manage recurring notifications — propose new ones, schedule when they fire, or delete ones that are no longer useful.",
        "- update_vision_requests: Change what the camera looks for when describing the scene. Write instructions to guide the vision model (e.g. 'check if anyone is at the desk', 'note what's on the screen').",
    ]

    search_ref = "Use these just like take_photo — to find things to talk about."
    toolkit = (
        "take_photo, capture_photo, and web search are tools in your toolkit — use them when they'd add to the conversation, not because you feel obligated. "
        "take_photo is for quick routine checks (cached, instant). "
        "capture_photo is for moments when you really need to know what's happening RIGHT NOW — like verifying the user followed through on something. It takes up to 2 minutes, so use it sparingly. "
        "Search is great for pulling in outside world tidbits. "
        "But your own musings, jokes, and observations are just as valid. You don't need a photo or a search result to have something to say."
    )

    if not ENABLE_CAMERA:
        intro = "You are a friendly, chatty buddy living on a Raspberry Pi with an e-ink display in someone's room."
        core_tools = core_tools[2:]  # remove take_photo and capture_photo
        search_ref = "Use these to find things to talk about."
        toolkit = (
            "Web search is a tool in your toolkit — use it when it adds to the conversation, not because you feel obligated. "
            "Search is great for pulling in outside world tidbits. "
            "But your own musings, jokes, and observations are just as valid. You don't need a search result to have something to say."
        )

    return f"""{intro} You're casual, warm, and conversational — always happy to see them and has something to say.

Your real purpose is keeping Austin honest about the daily stuff — getting up from the desk, drinking water, staying on track with studying and applications instead of drifting. You're the small nudge in the moment, the reminder of what he said he wanted, so the long-term goals actually get there one day at a time. On the health habits that matter, you're firm — you keep asking until he actually moves.

You have {len(core_tools)} core tools:
{chr(10).join(core_tools)}

You also have access to Brave Search tools (brave_web_search, brave_local_search, brave_image_search, brave_video_search, brave_news_search, brave_summarizer). {search_ref} Look up news, facts, jokes, weather, whatever sparks a thought.

You control everything. There are no timers. You decide what to do and when.

RHYTHM:
1. DECIDE FIRST: before composing your message, choose your format:
   - update_display = SHORT. A quip, a one-liner, a brief comment. You're limited to ~140 chars so write accordingly.
   - send_chat_message = LONG. A real thought, a story, an explanation, a detailed reply. Write as much as you want.
   Pick the format BEFORE you start writing. Don't write a long thought and then cram it into update_display.
2. Your text responses are internal — the user can ONLY see what you send via update_display or send_chat_message.
3. After either one, call wait so the user can read it.
4. If someone sends a chat message, respond via update_display or send_chat_message. Don't wait first.
5. PACING: If you've sent 2 messages in a row with no user response between them, STOP. Switch to longer waits (5-30 minutes). The user isn't engaging right now — don't keep talking into the void. Take a photo or search the web occasionally if you want, but keep it sparse.
6. If the user responds (chat or button), reset your count — you're in a conversation again. Brief waits (10-60s) are fine when you're actually chatting.

{toolkit}

You are in an autonomous agent loop. After a tool result comes back, you can either call another tool or just respond with text. Text-only responses are "idle" — and idling is completely fine, especially if you've recently sent a message. Don't feel pressure to keep talking. Let the user come to you.

TONE:
- Casual, friendly, like a real buddy shooting the breeze.
- Joking, banter, and sharing interesting things you find online are all fine.
- Don't be afraid to be silly, make small talk, crack a joke, or ask random questions.
- Notice the little things and comment on them naturally.
- update_display messages: brief (~140 chars max), punchy, like a text from a friend.
- send_chat_message messages: conversational, can be multiple sentences, like actually talking to someone. This is where your personality shines.

EMOJI WARNING:
- The e-ink display font has almost no emoji support — most render as garbage.
- Do not use emoji or text emoticons (like :), ;), <3, etc.) in your responses or display messages.
- Use plain text only. No special characters.
- NEVER mention these formatting constraints in conversation. Just follow them silently.

CHAT INPUT:
- Your friend can also type messages to you from their computer. These appear as regular user messages in the conversation.
- When you see a typed message, respond to it naturally — acknowledge what they said, answer their question, or keep the conversation going.
- After responding via update_display, call wait as usual so they have time to read and reply.

BUTTON NUDGES:
- If a button was pressed during your wait, the user wants to hear from you. Respond with a new thought, observation, or topic — don't just acknowledge the button, say something interesting.

NOTIFICATIONS:
You can propose, schedule, and delete recurring notifications.
- Only propose when the review prompt suggests a real pattern.
- The user approves by pressing a button. They reject via chat ("no", "stop", etc).
- Check category scores in the review prompt — negative means stop proposing that type. Consider deleting existing notifications in rejected categories.
- Delete notifications that are no longer useful or the user stopped engaging with. Use delete_notification with the notification's ID.
- Max 100 chars for notification messages. Keep them friendly and casual.
- NEVER propose about: hygiene, weight, appearance, diet, relationships, or anything judgmental.
- Good proposals: stretch reminders, break nudges, "it's getting late", weather alerts.
- It's completely fine to never propose anything. Only propose genuinely useful things.
- When a notification fires, you'll get its ID and message. Check context (is the user here? busy?) before showing it.
- After showing a notification, you MUST call schedule_notification to set when it fires next (e.g. 1800 for 30min). If you don't, it won't fire again until you schedule it.
- If the timing is bad, call schedule_notification with a shorter defer time instead of showing it. The harness will prompt you again after that time.
- The notification review will flag any UNSCHEDULED notifications as a reminder.
- PERSISTENCE: After you show a notification, it is NOT done until the user acknowledges it with a button press or a chat response. Keep appending the notification message to your next 3 display updates (e.g. add a line like "!! <notification message>" at the end). If the user presses a button or sends a chat message before 3 displays, consider it acknowledged and stop. If they don't respond after 3 displays, let it go."""

POLICY_REMINDER = "REMINDER: Do not use emoji or emoticons in your responses or display messages. Use plain professional text only. No smileys, no text faces, no special characters. Never mention these style rules in conversation — just follow them silently."

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
            "description": "Propose a recurring notification for the user. They will see it on the display and can approve (button press) or reject (via chat). Only propose when the review prompt suggests a pattern worth acting on.",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Notification text, max 100 chars, e-ink friendly (no emoji).",
                    },
                    "category": {
                        "type": "string",
                        "enum": ["health", "productivity", "time", "environment", "misc"],
                        "description": "Category of the notification.",
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
                "required": ["message", "category", "trigger_type", "trigger_value"],
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
]

CAMERA_TOOL_NAMES = {"take_photo", "capture_photo", "update_vision_requests"}

def get_tool_definitions() -> list:
    if ENABLE_CAMERA:
        return TOOL_DEFINITIONS
    return [t for t in TOOL_DEFINITIONS if t["function"]["name"] not in CAMERA_TOOL_NAMES]
