"""Cross-platform camera capture for Raspberry Pi and macOS webcams."""

import base64
import io
import sys
import threading

import numpy as np
from PIL import Image

from config import (
    CAMERA_BACKEND,
    CAMERA_DEVICE_INDEX,
    CAMERA_HEIGHT,
    CAMERA_IMAGE_MAX_WIDTH,
    CAMERA_ROTATION,
    CAMERA_WIDTH,
    JPEG_QUALITY,
)


LORES_SIZE = (160, 120)
LANCZOS = getattr(Image, "Resampling", Image).LANCZOS


def _encode_image(img: Image.Image) -> tuple[bytes, str]:
    """Apply mounting correction and encode the frame for the vision API."""
    if CAMERA_ROTATION % 360:
        img = img.rotate(CAMERA_ROTATION, expand=True)
    if CAMERA_IMAGE_MAX_WIDTH > 0 and img.width > CAMERA_IMAGE_MAX_WIDTH:
        ratio = CAMERA_IMAGE_MAX_WIDTH / img.width
        img = img.resize(
            (CAMERA_IMAGE_MAX_WIDTH, max(1, int(img.height * ratio))),
            LANCZOS,
        )
    stream = io.BytesIO()
    img.convert("RGB").save(stream, format="JPEG", quality=JPEG_QUALITY)
    jpeg_bytes = stream.getvalue()
    b64 = base64.b64encode(jpeg_bytes).decode("ascii")
    return jpeg_bytes, f"data:image/jpeg;base64,{b64}"


class _PiCamera:
    """IMX708/libcamera implementation used by the existing Pi deployment."""

    def __init__(self):
        try:
            from picamera2 import Picamera2
        except ImportError as exc:
            raise RuntimeError(
                "picamera2 camera backend requested but picamera2 is not installed"
            ) from exc

        self.picam = Picamera2()
        self._lock = threading.Lock()
        config = self.picam.create_still_configuration(
            main={"size": (CAMERA_WIDTH, CAMERA_HEIGHT)},
            lores={"size": LORES_SIZE, "format": "YUV420"},
        )
        self.picam.configure(config)
        self.picam.start()

    def capture(self) -> tuple[bytes, str]:
        with self._lock:
            request = self.picam.capture_request()
            try:
                return _encode_image(Image.fromarray(request.make_array("main")))
            finally:
                request.release()

    def capture_lores(self) -> np.ndarray:
        with self._lock:
            request = self.picam.capture_request()
            try:
                arr = request.make_array("lores")
                # YUV420: the Y plane is the first height rows.
                return arr[:LORES_SIZE[1], :LORES_SIZE[0]].astype(np.float32)
            finally:
                request.release()

    def close(self):
        self.picam.stop()


class _OpenCVCamera:
    """Webcam implementation, using AVFoundation on macOS."""

    def __init__(self):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "opencv camera backend requested but opencv-python is not installed"
            ) from exc

        self.cv2 = cv2
        self._lock = threading.Lock()
        api = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(CAMERA_DEVICE_INDEX, api)
        if not self.cap.isOpened():
            self.cap.release()
            raise RuntimeError(
                f"Could not open webcam index {CAMERA_DEVICE_INDEX}. On macOS, "
                "grant camera access to the terminal or service running AI Roommate."
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

    def _read_frame(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Could not read webcam index {CAMERA_DEVICE_INDEX}")
        return frame

    def capture(self) -> tuple[bytes, str]:
        with self._lock:
            frame = self._read_frame()
            rgb = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
            return _encode_image(Image.fromarray(rgb))

    def capture_lores(self) -> np.ndarray:
        with self._lock:
            frame = self._read_frame()
            gray = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2GRAY)
            gray = self.cv2.resize(gray, LORES_SIZE, interpolation=self.cv2.INTER_AREA)
            return gray.astype(np.float32)

    def close(self):
        self.cap.release()


def _select_backend() -> str:
    if CAMERA_BACKEND not in ("auto", "picamera2", "opencv"):
        raise ValueError(
            "CAMERA_BACKEND must be one of: auto, picamera2, opencv"
        )
    if CAMERA_BACKEND != "auto":
        return CAMERA_BACKEND
    if sys.platform == "darwin":
        return "opencv"
    try:
        import picamera2  # noqa: F401
    except ImportError:
        return "opencv"
    return "picamera2"


class Camera:
    """Stable camera facade used by the orchestrator on both platforms."""

    def __init__(self):
        backend = _select_backend()
        self.backend = backend
        self._impl = _PiCamera() if backend == "picamera2" else _OpenCVCamera()

    def capture(self) -> tuple[bytes, str]:
        return self._impl.capture()

    def capture_lores(self) -> np.ndarray:
        return self._impl.capture_lores()

    def close(self):
        self._impl.close()
