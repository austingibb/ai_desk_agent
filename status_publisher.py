"""AWS status publisher — uploads the caffeine/presence feed to S3 for aarg.dev.

Runs as a daemon thread on the Pi 5. Publishes when a drink is logged (trigger),
when the active state flips, and on a periodic heartbeat. Local state stays
authoritative — upload failures retry with backoff and never reach the agent loop.

Published contract (the website depends on this exact shape):
    {"active": true, "drinks": [{"t": 1783600000000, "mg": 95}, ...]}
"""

import json
import threading
import time
from config import (
    ENABLE_STATUS_PUBLISH,
    STATUS_S3_BUCKET,
    STATUS_S3_KEY,
    STATUS_PUBLISH_INTERVAL,
)
from logger import info


class StatusPublisher:
    def __init__(self, drink_store, active_tracker):
        self.drink_store = drink_store
        self.active_tracker = active_tracker
        self.running = False
        self._event = threading.Event()
        self._client = None
        self._last_published_active = None
        self._last_logged_payload = None
        self._failures = 0

    @property
    def enabled(self) -> bool:
        return ENABLE_STATUS_PUBLISH and bool(STATUS_S3_BUCKET)

    def trigger(self):
        """Request an immediate publish (e.g. right after a drink is logged)."""
        self._event.set()

    def start(self):
        if not self.enabled:
            info("[STATUS] Publisher disabled (set STATUS_S3_BUCKET + AWS creds in .env to enable)")
            return
        self.running = True
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()
        info(f"[STATUS] Publisher started -> s3://{STATUS_S3_BUCKET}/{STATUS_S3_KEY} "
             f"(heartbeat {STATUS_PUBLISH_INTERVAL}s)")

    def stop(self):
        self.running = False
        self._event.set()

    def _get_client(self):
        if self._client is None:
            import boto3
            self._client = boto3.client("s3")
        return self._client

    def _loop(self):
        last_publish = 0.0
        while self.running:
            active = self.active_tracker.is_active()
            due = (
                self._event.is_set()
                or active != self._last_published_active
                or time.time() - last_publish >= STATUS_PUBLISH_INTERVAL
            )
            if due:
                self._event.clear()
                if self._publish(active):
                    last_publish = time.time()
                    self._last_published_active = active
                    self._failures = 0
                else:
                    self._failures += 1
                    delay = min(5 * (2 ** (self._failures - 1)), 300)
                    info(f"[STATUS] Publish failed ({self._failures}x), retrying in {delay}s")
                    self._sleep(delay)
                    continue
            self._sleep(1)
        info("[STATUS] Publisher stopped")

    def _sleep(self, seconds):
        for _ in range(int(seconds)):
            if not self.running or self._event.is_set():
                return
            time.sleep(1)

    def _publish(self, active: bool) -> bool:
        payload = {
            "active": bool(active),
            "drinks": self.drink_store.get_feed_drinks(),
        }
        try:
            self._get_client().put_object(
                Bucket=STATUS_S3_BUCKET,
                Key=STATUS_S3_KEY,
                Body=json.dumps(payload).encode(),
                ContentType="application/json",
                CacheControl="no-store",
            )
            # Heartbeats republish the same payload — only log when it changes
            summary = (payload["active"], len(payload["drinks"]))
            if summary != self._last_logged_payload:
                self._last_logged_payload = summary
                info(f"[STATUS] Published: active={payload['active']}, {len(payload['drinks'])} drinks")
            return True
        except Exception as e:
            info(f"[STATUS] Upload error: {e}")
            return False
