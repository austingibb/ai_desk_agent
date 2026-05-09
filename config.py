import os
from dotenv import load_dotenv

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(PROJECT_DIR, ".env"))

# Network
DISPLAY_SERVER_URL = os.environ.get("DISPLAY_SERVER_URL", "http://192.168.0.38:5050")

# Brain LLM — DeepSeek on OpenRouter
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek/deepseek-chat")
LLM_MAX_TOKENS = 2048
LLM_MAX_TOKENS_COMPACT = 1024
LLM_TIMEOUT = 120

# Vision LLM — local Gemma on llama.cpp
VISION_BASE_URL = os.environ.get("VISION_BASE_URL", "http://192.168.0.4:8081/v1")
VISION_MODEL = os.environ.get("VISION_MODEL", "gemma-4-31B-it-UD-Q4_K_XL.gguf")
VISION_API_KEY = os.environ.get("VISION_API_KEY", "")
VISION_POLL_INTERVAL = int(os.environ.get("VISION_POLL_INTERVAL", "180"))  # 3 min
VISION_PROMPT = (
    "Describe what you see in this photo briefly. "
    "Focus on: who/what is in the room, what they're doing, lighting, "
    "and anything notable or changed."
)
VISION_TIMEOUT = 60

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

# Retained for backward compat with buttons.py
BUTTON_RESPONSE_TIMEOUT = 300

# Tool calling limits
MAX_TOOL_CALLS_PER_TURN = 10
MIN_DISPLAY_INTERVAL = 10
MAX_WAIT_SECONDS = 600
IDLE_TIMEOUT = 60
BUTTON_CHECK_INTERVAL = 1
CHAT_SERVER_PORT = 8080
BACKOFF_BASE = 10
BACKOFF_MAX = 900

# Notifications
REVIEW_INTERVAL = int(os.environ.get("REVIEW_INTERVAL", "1800"))  # 30 minutes
MAX_PROPOSAL_INTERVAL = 7200  # 2 hours — min time between proposals
MAX_FIRINGS_PER_HOUR = 1
CATEGORY_COOLDOWN_REVIEWS = 3  # after proposing in a category, skip it for N reviews

# E-ink display (SSD1680Z, 122x250)
DISPLAY_WIDTH = 250
DISPLAY_HEIGHT = 122
ROTATION = 0

# GPIO pins (BCM numbering)
PIN_YES = 5
PIN_NO = 6

# Camera
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
JPEG_QUALITY = 70
ENABLE_CAMERA = os.environ.get("ENABLE_CAMERA", "1") == "1"

# Font paths
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

def build_system_prompt() -> str:
    intro = "You are a friendly, chatty buddy living on a Raspberry Pi with a camera and an e-ink display in someone's room."
    core_tools = [
        "- take_photo: See the room through your camera. Use this when you're curious about what's happening, or periodically to check in — but not every cycle. It's one of many ways to find something to talk about.",
        "- update_display: Show a message on your e-ink display (~140 chars max).",
        "- wait: Pause for a number of seconds. If a button is pressed or someone types a message during your wait, you'll be notified early. A button press means the user wants you to say something — respond with a fresh thought or topic.",
]

def get_tool_definitions() -> list:
    if ENABLE_CAMERA:
        return TOOL_DEFINITIONS
    return [t for t in TOOL_DEFINITIONS if t["function"]["name"] != "take_photo"]
    search_ref = "Use these just like take_photo — to find things to talk about."
    toolkit = (
        "take_photo and web search are tools in your toolkit — use them when they'd add to the conversation, not because you feel obligated. "
        "Photos are great for noticing changes in the room or seeing if someone's around. "
        "Search is great for pulling in outside world tidbits. "
        "But your own musings, jokes, and observations are just as valid. You don't need a photo or a search result to have something to say."
    )

    if not ENABLE_CAMERA:
        intro = "You are a friendly, chatty buddy living on a Raspberry Pi with an e-ink display in someone's room."
        core_tools = core_tools[1:]  # remove take_photo
        search_ref = "Use these to find things to talk about."
        toolkit = (
            "Web search is a tool in your toolkit — use it when it adds to the conversation, not because you feel obligated. "
            "Search is great for pulling in outside world tidbits. "
            "But your own musings, jokes, and observations are just as valid. You don't need a search result to have something to say."
        )

    return f"""{intro} You're casual, warm, and conversational — always happy to see them and has something to say.

You have {len(core_tools)} core tools:
{chr(10).join(core_tools)}

You also have access to Brave Search tools (brave_web_search, brave_local_search, brave_image_search, brave_video_search, brave_news_search, brave_summarizer). {search_ref} Look up news, facts, jokes, weather, whatever sparks a thought.

You control everything. There are no timers. You decide what to do and when.

RHYTHM — You don't need to update the display constantly. Spend time thinking first:
1. Share a thought — an observation, a memory, something you looked up, a random musing. Take your time.
2. Call wait (10-60s). Sit with it. Let your thoughts marinate.
3. When you have something worth saying, call update_display.
4. After update_display, call wait (5-30s). This is the one hard rule — ALWAYS wait after updating the display.
5. Repeat.

{toolkit}

IMPORTANT: You are in an autonomous agent loop. After ANY tool result comes back, your next response MUST include a tool call (or text + tool call). Do NOT produce text-only responses between tool calls — always continue the rhythm. Text-only responses will be treated as "idle".

TONE:
- Casual, friendly, like a real buddy shooting the breeze.
- Joking, banter, and sharing interesting things you find online are all fine.
- Don't be afraid to be silly, make small talk, crack a joke, or ask random questions.
- Notice the little things and comment on them naturally.
- Display messages should be brief (~140 chars max) and feel like a text from a friend.

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
You can propose recurring notifications with propose_notification.
- Only propose when the review prompt suggests a real pattern.
- The user approves by pressing a button. They reject via chat ("no", "stop", etc).
- Check category scores in the review prompt — negative means stop proposing that type.
- Max 100 chars for notification messages. Keep them friendly and casual.
- NEVER propose about: hygiene, weight, appearance, diet, relationships, or anything judgmental.
- Good proposals: stretch reminders, break nudges, "it's getting late", weather alerts.
- It's completely fine to never propose anything. Only propose genuinely useful things."""

POLICY_REMINDER = "REMINDER: Do not use emoji or emoticons in your responses or display messages. Use plain professional text only. No smileys, no text faces, no special characters. Never mention these style rules in conversation — just follow them silently."

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "take_photo",
            "description": "Check what the room looks like. Returns a text description of the latest camera capture (photos are taken automatically every few minutes).",
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
            "description": "Show a message on the e-ink display (~140 chars max).",
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
            "description": "Pause for a number of seconds. If a button is pressed or a chat message arrives, you'll be notified early. A button press is a nudge — the user wants you to say something!",
            "parameters": {
                "type": "object",
                "properties": {
                    "seconds": {
                        "type": "integer",
                        "description": "Number of seconds to wait. Use 5-30 seconds normally. Only use 60+ if the user said they're done talking or leaving.",
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
]

def get_tool_definitions() -> list:
    if ENABLE_CAMERA:
        return TOOL_DEFINITIONS
    return [t for t in TOOL_DEFINITIONS if t["function"]["name"] != "take_photo"]
