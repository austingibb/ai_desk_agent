"""E-ink display driver for Adafruit SSD1680Z (122x250, SPI)."""

import digitalio
import busio
import board
from PIL import Image, ImageDraw, ImageFont
from adafruit_epd.ssd1680 import Adafruit_SSD1680Z
from adafruit_epd.epd import Adafruit_EPD
from config import DISPLAY_WIDTH, DISPLAY_HEIGHT, ROTATION, FONT_BOLD, FONT_REGULAR


class Display:
    def __init__(self):
        spi = busio.SPI(board.SCK, MOSI=board.MOSI, MISO=board.MISO)
        ecs = digitalio.DigitalInOut(board.CE0)
        dc = digitalio.DigitalInOut(board.D22)
        rst = digitalio.DigitalInOut(board.D27)
        busy = digitalio.DigitalInOut(board.D17)

        self.epd = Adafruit_SSD1680Z(
            DISPLAY_HEIGHT,
            DISPLAY_WIDTH,
            spi,
            cs_pin=ecs,
            dc_pin=dc,
            sramcs_pin=None,
            rst_pin=rst,
            busy_pin=busy,
        )
        self.epd.rotation = ROTATION

        self.width = self.epd.width
        self.height = self.epd.height

        self.font_bold_lg = self._load_font(FONT_BOLD, 22)
        self.font_bold_md = self._load_font(FONT_BOLD, 16)
        self.font_regular = self._load_font(FONT_REGULAR, 14)
        self.font_small = self._load_font(FONT_REGULAR, 10)

        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)

    def _load_font(self, path: str, size: int) -> ImageFont.FreeTypeFont:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            return ImageFont.load_default()

    def _wrap_text(self, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list:
        words = text.split()
        lines = []
        current = ""
        for word in words:
            test = f"{current} {word}".strip()
            bbox = font.getbbox(test)
            if bbox[2] <= max_width:
                current = test
            else:
                if current:
                    lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines if lines else [text]

    def show_text(self, text: str, question: str = None):
        self.epd.fill(Adafruit_EPD.WHITE)
        image = Image.new("RGB", (self.width, self.height), color=self.WHITE)
        draw = ImageDraw.Draw(image)

        margin = 8
        max_width = self.width - margin * 2

        lines = self._wrap_text(text, self.font_regular, max_width)
        y = margin
        line_height = self.font_regular.getbbox("Tg")[3] + 2
        for line in lines:
            if y + line_height > self.height - 30:
                break
            draw.text((margin, y), line, font=self.font_regular, fill=self.BLACK)
            y += line_height

        if question:
            y = self.height - 24
            q_prefix = f"[YES]  {question}  [NO]"
            draw.text((margin, y), q_prefix, font=self.font_small, fill=self.BLACK)

        self.epd.image(image)
        self.epd.display()

    def show_booting(self):
        self.show_text("Waking up...")
