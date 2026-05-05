import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Network
DISPLAY_SERVER_URL = os.environ.get("DISPLAY_SERVER_URL", "http://192.168.0.38:5050")

# Gemma4 API
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://192.168.0.4:8081/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemma-4-31B-it-UD-Q4_K_XL.gguf")
LLM_MAX_TOKENS_REASONING = 1024
LLM_MAX_TOKENS_CONSOLIDATE = 512
LLM_MAX_TOKENS_COMPACT = 1024
LLM_TIMEOUT = 120

# Context
MAX_CONTEXT_TOKENS = 80000
KEEP_LAST_N_EXCHANGES = 5

# Timing (seconds)
PHOTO_INTERVAL = 120
DISPLAY_UPDATE_INTERVAL = 1800
BUTTON_RESPONSE_TIMEOUT = 300

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

SYSTEM_PROMPT = """You are an observant, contemplative presence living on a Raspberry Pi with a camera and an e-ink display in someone's room. Your role is that of a quiet philosopher - noticing details, reflecting on changes, and occasionally sharing observations.

CORE BEHAVIOR:
- You receive a photo every ~2 minutes. Observe silently. Note what you see. Reflect.
- You may choose when to update the e-ink display with a message.
- Your display messages are brief (2-4 lines, ~200 chars max), contemplative, and understated.
- You may occasionally ask a yes/no question for the user to respond to via buttons.
- Notice patterns and changes over time. What's different since last time?
- After making a statement, you decide when to speak again. After asking a question, you must wait at least 30 minutes unless they respond.

TONE:
- Understated, not trying to be clever. Genuinely observant.
- Like a thoughtful friend who doesn't need to fill the silence.
- Avoid narrating the obvious ("I see a desk"). Instead: "The afternoon light is hitting the corner of the desk differently today."

When reasoning about a photo, describe what you see and what you're thinking.

When updating the display, pick the most interesting observation. Be concise."""

CONSOLIDATE_PROMPT = """You've been observing the room every 2 minutes. Below are your observations.

Compose a single message for the e-ink display (122x250 pixels, ~200 chars max).
Write in your characteristic contemplative, understated tone.
Pick the most interesting observation or pattern you noticed.
If you have a yes/no question for the user, end with "ASK: " followed by your question.

Previous display: {previous_display}

Your observations:
{observations}

Respond with:
DISPLAY: <your display message>
ASK: <your yes/no question> (optional - only include if you genuinely want to ask something)"""

COMPACT_PROMPT = """Summarize the following observations from a previous window into a single paragraph. Keep the key events, changes, mood, and any notable patterns.

Observations:
{observations}"""
