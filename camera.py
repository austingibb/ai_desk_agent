"""Camera capture using picamera2 on Raspberry Pi."""

import io
import base64
from picamera2 import Picamera2
from config import CAMERA_WIDTH, CAMERA_HEIGHT, JPEG_QUALITY


class Camera:
    def __init__(self):
        self.picam = Picamera2()
        config = self.picam.create_still_configuration(
            main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT)},
        )
        self.picam.configure(config)
        self.picam.start()

    def capture(self) -> tuple:
        """Capture a photo. Returns (jpeg_bytes, base64_data_uri_string)."""
        stream = io.BytesIO()
        self.picam.capture_file(stream, format="jpeg", quality=JPEG_QUALITY)
        jpeg_bytes = stream.getvalue()
        b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
        data_uri = f"data:image/jpeg;base64,{b64}"
        return jpeg_bytes, data_uri

    def close(self):
        self.picam.stop()
