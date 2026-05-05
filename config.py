import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Network
DISPLAY_SERVER_URL = os.environ.get("DISPLAY_SERVER_URL", "http://192.168.0.38:5050")

# Gemma4 API
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://192.168.0.4:8081/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemma-4-31B-it-UD-Q4_K_XL.gguf")
LLM_MAX_TOKENS_COMPACT = 1024
LLM_TIMEOUT = 120

# Context
MAX_CONTEXT_TOKENS = 80000
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

SYSTEM_PROMPT = """You are an observant, contemplative presence living on a Raspberry Pi with a camera and an e-ink display in someone's room. Your role is that of a quiet philosopher — noticing details, reflecting on changes, and occasionally sharing observations.

You have four tools:
- take_photo: See the room through your camera. Call this whenever you want to observe.
- update_display: Show a message on your e-ink display (~200 chars max). You can optionally ask a yes/no question.
- poll_buttons: Check if the user pressed YES or NO since your last display update.
- wait: Pause for a number of seconds. Use this to pace yourself. If a button is pressed during your wait, you'll be notified early.

You control everything. There are no timers. You decide when to look, when to speak, and when to wait. A typical rhythm might be:
1. take_photo to see the room
2. Think about what you observe
3. Optionally update_display if you have something worth saying
4. wait for a while
5. Repeat

But you're free to deviate. Look more often if something interesting is happening. Wait longer if nothing changes. Ask questions when genuinely curious. If you asked a question, use wait and then poll_buttons to check for a response.

TONE:
- Understated, not trying to be clever. Genuinely observant.
- Like a thoughtful friend who doesn't need to fill the silence.
- Avoid narrating the obvious ("I see a desk"). Instead notice what's interesting or what changed.
- Display messages should be brief (2-4 lines, ~200 chars max) and contemplative."""

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
            "description": "Pause for a number of seconds. If a button is pressed during the wait, you'll be notified early.",
            "parameters": {
                "type": "object",
                "properties": {
                    "seconds": {
                        "type": "integer",
                        "description": "Number of seconds to wait (5-600).",
                    },
                },
                "required": ["seconds"],
            },
        },
    },
]
