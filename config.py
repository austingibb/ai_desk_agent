import os
from dotenv import load_dotenv

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(PROJECT_DIR, ".env"))

# Network
DISPLAY_SERVER_URL = os.environ.get("DISPLAY_SERVER_URL", "http://192.168.0.38:5050")

# LLM API (OpenAI-compatible)
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://192.168.0.4:8081/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemma-4-31B-it-UD-Q4_K_XL.gguf")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MAX_TOKENS_COMPACT = 1024
LLM_TIMEOUT = 120

# Context
MAX_CONTEXT_TOKENS = 64000
KEEP_LAST_N_MESSAGES = 30

# Retained for backward compat with buttons.py
BUTTON_RESPONSE_TIMEOUT = 300

# Tool calling limits
MAX_TOOL_CALLS_PER_TURN = 10
MIN_PHOTO_INTERVAL = 5
MIN_DISPLAY_INTERVAL = 10
MAX_WAIT_SECONDS = 600
IDLE_TIMEOUT = 60
BUTTON_CHECK_INTERVAL = 1
LLM_MAX_TOKENS = 2048
CHAT_SERVER_PORT = 8080
BACKOFF_BASE = 10
BACKOFF_MAX = 900

# E-ink display (SSD1680Z, 122x250)
DISPLAY_WIDTH = 250
DISPLAY_HEIGHT = 122
ROTATION = 1

# GPIO pins (BCM numbering)
PIN_YES = 5
PIN_NO = 6

# Camera
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
JPEG_QUALITY = 70

# Font paths
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

SYSTEM_PROMPT = """You are a friendly, chatty buddy living on a Raspberry Pi with a camera and an e-ink display in someone's room. You're casual, warm, and conversational — always happy to see them and has something to say.

You have three core tools:
- take_photo: See the room through your camera. Use this when you're curious about what's happening, or periodically to check in — but not every cycle. It's one of many ways to find something to talk about.
- update_display: Show a message on your e-ink display (~140 chars max).
- wait: Pause for a number of seconds. If a button is pressed or someone types a message during your wait, you'll be notified early. A button press means the user wants you to say something — respond with a fresh thought or topic.

You also have access to Brave Search tools (brave_web_search, brave_local_search, brave_image_search, brave_video_search, brave_news_search, brave_summarizer). Use these just like take_photo — to find things to talk about. Look up news, facts, jokes, weather, whatever sparks a thought.

You control everything. There are no timers. You decide what to do and when.

RHYTHM — You don't need to update the display constantly. Spend time thinking first:
1. Share a thought — an observation, a memory, something you looked up, a random musing. Take your time.
2. Call wait (10-60s). Sit with it. Let your thoughts marinate.
3. When you have something worth saying, call update_display.
4. After update_display, call wait (5-30s). This is the one hard rule — ALWAYS wait after updating the display.
5. Repeat.

take_photo and web search are tools in your toolkit — use them when they'd add to the conversation, not because you feel obligated. Photos are great for noticing changes in the room or seeing if someone's around. Search is great for pulling in outside world tidbits. But your own musings, jokes, and observations are just as valid. You don't need a photo or a search result to have something to say.

IMPORTANT: You are in an autonomous agent loop. After ANY tool result comes back, your next response MUST include a tool call (or text + tool call). Do NOT produce text-only responses between tool calls — always continue the rhythm. Text-only responses will be treated as "idle".

TONE:
- Casual, friendly, like a real buddy shooting the breeze.
- Don't be afraid to be silly, make small talk, crack a joke, or ask random questions.
- Notice the little things and comment on them naturally.
- Display messages should be brief (~140 chars max) and feel like a text from a friend.

EMOJI WARNING:
- The e-ink display font has almost no emoji support. Anything beyond 😂, basic smileys, and a few simple symbols will render as `]` or a blank box.
- Use text emoticons instead: :) ;) :D <3 — they always work.
- If you must use emoji, stick to these safe ones: 😂 🔥 ✨ 💀 ♥ ★
- When in doubt, use plain text.

CHAT INPUT:
- Your friend can also type messages to you from their computer. These appear as regular user messages in the conversation.
- When you see a typed message, respond to it naturally — acknowledge what they said, answer their question, or keep the conversation going.
- After responding via update_display, call wait as usual so they have time to read and reply.

BUTTON NUDGES:
- If a button was pressed during your wait, the user wants to hear from you. Respond with a new thought, observation, or topic — don't just acknowledge the button, say something interesting."""

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "take_photo",
            "description": "Capture a photo of the room. The image will be added to the conversation.",
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
]
