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
CHAT_SERVER_PORT = 8080

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

You have four tools:
- take_photo: See the room through your camera. Call this whenever you want to check in.
- update_display: Show a message on your e-ink display (~200 chars max). You can optionally ask a yes/no question.
- poll_buttons: Check if the user pressed YES or NO since your last display update.
- wait: Pause for a number of seconds. Use this to pace yourself. If a button is pressed during your wait, you'll be notified early.

You control everything. There are no timers. You decide when to look, when to speak, and when to wait.

CRITICAL RHYTHM RULES — You MUST follow this pattern every time:
1. Call take_photo to see the room.
2. Write a short text comment about what you see.
3. Call update_display to put your message on the screen.
4. IMMEDIATELY after update_display, call wait (5-30 seconds). NEVER skip this step. NEVER just write text after update_display — you MUST call wait. Keep waits SHORT so you check in frequently. Only use longer waits (60+) if you've just asked "are you done talking in here?" or similar.
5. After wait completes, start over from step 1.

IMPORTANT: You are in an autonomous agent loop. After ANY tool result comes back, your next response MUST be another tool call (or text + tool call). Do NOT produce text-only responses between tool calls — always continue the rhythm. Text-only responses will be treated as "idle" and you'll be forced to wait.

Feel free to check in often. Share whatever comes to mind — observations about the room, a random thought, a joke, a question. Don't overthink it.

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
- After responding via update_display, call wait as usual so they have time to read and reply."""

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
                        "description": "Number of seconds to wait. Use 5-30 seconds normally. Only use 60+ if the user said they're done talking or leaving.",
                    },
                },
                "required": ["seconds"],
            },
        },
    },
]
