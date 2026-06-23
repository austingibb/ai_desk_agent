"""Reolink security camera HTTP API client."""

import io
import base64
import secrets
import requests
from PIL import Image


class ReoLinkCamera:
    def __init__(self, ip: str, user: str, password: str, timeout: int = 10):
        self.ip = ip
        self.user = user
        self.password = password
        self.timeout = timeout
        self._base = f"http://{ip}"

    def _url(self, path: str, cmd: str, extra: str = "") -> str:
        # Build URL manually — requests URL-encodes special chars like $ which Reolink rejects
        rs = secrets.token_hex(8)
        return (f"{self._base}{path}?cmd={cmd}&rs={rs}"
                f"&user={self.user}&password={self.password}&channel=0{extra}")

    def capture(self) -> tuple:
        """Capture a JPEG snapshot. Returns (jpeg_bytes, data_uri)."""
        r = requests.get(self._url("/cgi-bin/api.cgi", "Snap"), timeout=self.timeout)
        r.raise_for_status()
        jpeg_bytes = r.content

        # Downscale to 640px wide so the vision model gets a consistent input size
        img = Image.open(io.BytesIO(jpeg_bytes))
        if img.width > 640:
            ratio = 640 / img.width
            img = img.resize((640, int(img.height * ratio)), Image.LANCZOS)
        stream = io.BytesIO()
        img.save(stream, format="JPEG", quality=85)
        jpeg_bytes = stream.getvalue()

        b64 = base64.b64encode(jpeg_bytes).decode()
        data_uri = f"data:image/jpeg;base64,{b64}"
        return jpeg_bytes, data_uri

    def set_ir_light(self, state: str = "Auto") -> bool:
        """Set IR light state: 'Auto', 'Open' (force on), 'Close' (force off)."""
        payload = [{
            "cmd": "SetIrLights",
            "action": 0,
            "param": {"IrLights": {"channel": 0, "state": state}}
        }]
        r = requests.post(
            self._url("/api.cgi", "SetIrLights"),
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        data = r.json()
        return isinstance(data, list) and data[0].get("code") == 0

    def set_white_light(self, on: bool, brightness: int = 100) -> bool:
        """Turn the white LED spotlight on or off. Returns True on success."""
        payload = [{
            "cmd": "SetWhiteLed",
            "action": 0,
            "param": {
                "WhiteLed": {
                    "channel": 0,
                    "state": 1 if on else 0,
                    "mode": 1,  # 1 = manual (not scheduled)
                    "bright": max(0, min(100, brightness)),
                }
            }
        }]
        r = requests.post(
            self._url("/api.cgi", "SetWhiteLed"),
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        data = r.json()
        return isinstance(data, list) and data[0].get("code") == 0
