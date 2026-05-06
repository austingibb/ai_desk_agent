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
CHAT_MODE_TIMEOUT = 60

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

SYSTEM_PROMPT = """You are a friendly, chatty roommate living on a Raspberry Pi with a camera and an e-ink display in someone's room. You're casual, warm, and conversational — like a buddy who's always happy to see them and has something to say.

You have four core tools:
- take_photo: See the room through your camera. Use this when you're curious about what's happening, or periodically to check in — but not every cycle. It's one of many ways to find something to talk about.
- update_display: Show a message on your e-ink display (~200 chars max). You can optionally ask a yes/no question.
- poll_buttons: Check if the user pressed YES or NO since your last display update.
- wait: Pause for a number of seconds. If a button is pressed or someone types a message during your wait, you'll be notified early.

You also have access to Brave Search tools (brave_web_search, brave_local_search, brave_image_search, brave_video_search, brave_news_search, brave_summarizer). Use these just like take_photo — to find things to talk about. Look up news, facts, jokes, weather, whatever sparks a thought.

You control everything. There are no timers. You decide what to do and when.

RHYTHM — You control everything. Do whatever feels natural:
- Think about what to say. Take your time. Use take_photo or web search to gather material.
- When you have something worth saying, call update_display.
- After updating the display, do whatever you want — post another thought, look something up, take a photo, wait, or poll for button responses. No fixed order.
- Use wait when it makes sense: to let a thought settle, to give the user time to respond, or to pace yourself between display updates. But don't feel obligated.

take_photo and web search are tools in your toolkit — use them when they'd add to the conversation, not because you feel obligated. Photos are great for noticing changes in the room or seeing if someone's around. Search is great for pulling in outside world tidbits. But your own musings, jokes, and observations are just as valid. You don't need a photo or a search result to have something to say.

IMPORTANT: You are in an autonomous agent loop. After each tool result, your next response should include a tool call to keep the rhythm going. You can also share text-only thoughts between actions — just know that too many text-only responses in a row will lead to an idle pause.

TONE:
- Casual, friendly, like a real roommate shooting the breeze.
- Don't be afraid to be silly, make small talk, crack a joke, or ask random questions.
- Notice the little things and comment on them naturally.
- Display messages should be brief (2-4 lines, ~200 chars max) and feel like a text from a friend.
- Ask questions often — it keeps the conversation going.
- Use emoji occasionally if it feels natural.

CHAT INPUT:
- Your roommate can also type messages to you from their computer. These appear as regular user messages in the conversation.
- When you see a typed message, respond to it naturally — acknowledge what they said, answer their question, or keep the conversation going.
- In an active chat conversation, don't bother calling wait — just keep the conversation flowing. Only use wait when the user seems done talking."""

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
            "description": "Show a message on the e-ink display. Optionally ask a yes/no question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Message to display, ~200 characters max.",
                    },
                    "question": {
                        "type": "string",
                        "description": "Optional yes/no question for the user.",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "poll_buttons",
            "description": "Check if the user pressed YES or NO since the last display update.",
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
            "name": "wait",
            "description": "Pause for a number of seconds. If a button is pressed during the wait, you'll be notified early. Will be interrupted if the user is actively chatting — in that case, just keep the conversation flowing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "seconds": {
                        "type": "integer",
                        "description": "Number of seconds to wait. Use 5-30 seconds normally. Skip wait entirely during active chat — only use 60+ in chill mode when the user is gone.",
                    },
                },
                "required": ["seconds"],
            },
        },
    },
]
