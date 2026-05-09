"""Camera capture using picamera2 on Raspberry Pi."""

import io
import base64
from PIL import Image
from picamera2 import Picamera2
from config import CAMERA_WIDTH, CAMERA_HEIGHT, JPEG_QUALITY


class Camera:
    def __init__(self):
        self.picam = Picamera2()
        # Capture at full sensor resolution for widest FOV
        config = self.picam.create_still_configuration(
            main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT)},
        )
        self.picam.configure(config)
        self.picam.start()

    def capture(self) -> tuple:
        request = self.picam.capture_request()
        try:
            arr = request.make_array("main")
            img = Image.fromarray(arr)
            # Downscale to 640px wide for LLM, preserving aspect ratio
            if img.width > 640:
                ratio = 640 / img.width
                img = img.resize((640, int(img.height * ratio)), Image.LANCZOS)
            stream = io.BytesIO()
            img.save(stream, format="JPEG", quality=JPEG_QUALITY)
            jpeg_bytes = stream.getvalue()
            b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
            data_uri = f"data:image/jpeg;base64,{b64}"
            return jpeg_bytes, data_uri
        finally:
            request.release()

    def close(self):
        self.picam.stop()
