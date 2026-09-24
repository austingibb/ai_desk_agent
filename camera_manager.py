"""Camera-independent capture, image/description caches, and background updates."""

from dataclasses import dataclass, field
import math
import threading
import time
from typing import Protocol

from logger import info


class CaptureDevice(Protocol):
    def capture(self) -> tuple[bytes, str]: ...
    def close(self) -> None: ...


@dataclass
class CameraFeed:
    id: str
    device: CaptureDevice
    mode: str = "cache"
    interval: float = 180
    scene: dict | None = None
    jpeg: bytes | None = None
    last_motion: float = float("-inf")
    lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self):
        if self.mode not in {"cache", "poll_only"}:
            raise ValueError("Camera mode must be cache or poll_only")
        if not math.isfinite(self.interval) or self.interval <= 0:
            raise ValueError("Camera interval must be positive")


class CameraManager:
    def __init__(self, feeds, vision, *, on_scene=None, on_motion=None,
                 save_image=None, motion_interval=3, chill_timeout=300,
                 detector_factory=None):
        self.feeds = {}
        for feed in feeds:
            if feed.id in self.feeds:
                raise ValueError(f"Duplicate camera ID: {feed.id}")
            self.feeds[feed.id] = feed
        self.vision = vision
        self.on_scene = on_scene
        self.on_motion = on_motion
        self.save_image = save_image
        self.motion_interval = motion_interval
        self.chill_timeout = chill_timeout
        self.detector_factory = detector_factory
        # VisionClient has per-request metadata: keep it with its own image.
        self.job_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.threads = []

    def capture(self, camera_id, *, background=False):
        feed = self.feeds[camera_id]
        with self.job_lock:
            if self.stop_event.is_set():
                return None
            try:
                jpeg, uri = feed.device.capture()
                captured_at = time.time()
                source = f"{camera_id}_{'background' if background else 'on_demand'}"
                description = self.vision.describe(uri, source=source, captured_at=captured_at)
                if not description:
                    return None
                scene = {"camera_id": camera_id, "source": source,
                         "description": description, "timestamp": captured_at,
                         "request_commit": self.vision.last_request_commit}
                if feed.mode == "cache":
                    with feed.lock:
                        feed.jpeg = jpeg
                        feed.scene = scene
                if self.save_image:
                    self.save_image(jpeg, camera_id)
                if self.on_scene:
                    self.on_scene(scene, background)
                return scene
            except Exception as exc:
                info(f"[VISION:{camera_id}] Capture/describe failed: {exc}")
                return None

    def read(self, camera_id=None, *, fresh=False):
        camera_id = camera_id or next(iter(self.feeds), None)
        if camera_id not in self.feeds:
            return {"status": "error", "message": "Unknown or disabled camera",
                    "available_cameras": list(self.feeds)}
        feed = self.feeds[camera_id]
        with feed.lock:
            scene = feed.scene
        if fresh or feed.mode == "poll_only":
            scene = self.capture(camera_id)
        if not scene:
            return {"status": "error", "camera_id": camera_id,
                    "message": "No scene available yet; capture failed or cache is warming up"}
        return {"status": "ok", "camera_id": camera_id, "mode": feed.mode,
                "source": scene["source"], "description": scene["description"],
                "captured_at": time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(scene["timestamp"])),
                "age_seconds": max(0, int(time.time() - scene["timestamp"])),
                "vision_request_commit": scene["request_commit"]}

    @property
    def motion_active(self):
        now = time.monotonic()
        return any(now - feed.last_motion <= self.chill_timeout for feed in self.feeds.values())

    def start(self):
        if self.threads:
            return
        for feed in self.feeds.values():
            if feed.mode == "cache":
                thread = threading.Thread(target=self._loop, args=(feed,), daemon=True,
                                          name=f"camera-{feed.id}")
                self.threads.append(thread)
                thread.start()

    def _loop(self, feed):
        detector = None
        if self.detector_factory and callable(getattr(feed.device, "capture_lores", None)):
            detector = self.detector_factory()
        last_attempt = float('-inf')
        while not self.stop_event.is_set():
            now = time.monotonic()
            wake = False
            if detector:
                try:
                    from PIL import Image
                    gray = feed.device.capture_lores()
                    result = detector.check(Image.fromarray(gray.clip(0, 255).astype('uint8')))
                    if result["changed"]:
                        wake = now - feed.last_motion > self.chill_timeout
                        feed.last_motion = now
                        if self.on_motion:
                            self.on_motion()
                except Exception as exc:
                    info(f"[VISION:{feed.id}] Motion check failed: {exc}")
            # Refresh even without motion, and retry failures on the same cadence.
            if wake or now - last_attempt >= feed.interval:
                self.capture(feed.id, background=True)
                last_attempt = time.monotonic()
            self.stop_event.wait(self.motion_interval if detector else feed.interval)

    def close(self):
        self.stop_event.set()
        for thread in self.threads:
            thread.join()
        with self.job_lock:
            for feed in self.feeds.values():
                try:
                    feed.device.close()
                except Exception as exc:
                    info(f"[VISION:{feed.id}] Close failed: {exc}")
