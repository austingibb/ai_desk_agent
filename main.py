#!/usr/bin/env python3
"""AI E-Ink Roommate — main orchestrator. Runs on Pi 5 (192.168.0.39)."""

import time
import signal
import sys
import json
import requests
from config import (
    DISPLAY_SERVER_URL,
    SYSTEM_PROMPT,
    PHOTO_INTERVAL,
    DISPLAY_UPDATE_INTERVAL,
    BUTTON_RESPONSE_TIMEOUT,
    CAMERA_WIDTH,
    CAMERA_HEIGHT,
)
from context import Context
from camera import Camera
from ai_client import AIClient


def http_get(path: str, timeout: int = 5) -> dict:
    try:
        r = requests.get(f"{DISPLAY_SERVER_URL}{path}", timeout=timeout)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"[HTTP GET] {path}: {e}")
    return {}


def http_post(path: str, data: dict, timeout: int = 5):
    try:
        r = requests.post(f"{DISPLAY_SERVER_URL}{path}", json=data, timeout=timeout)
        return r.status_code == 200
    except Exception as e:
        print(f"[HTTP POST] {path}: {e}")
    return False


class Orchestrator:
    def __init__(self):
        self.ctx = Context()
        self.ctx.add_system(SYSTEM_PROMPT)

        print("Init camera...")
        self.camera = Camera()

        print("Init AI client...")
        self.ai = AIClient()

        self.last_photo_time = 0
        self.cooldown_until = time.monotonic()
        self.running = True

        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        print("\nShutting down...")
        self.running = False

    def _cooldown_passed(self) -> bool:
        return time.monotonic() >= self.cooldown_until

    def _set_cooldown(self, seconds: float):
        self.cooldown_until = time.monotonic() + seconds

    def run(self):
        print("Entering main loop.")
        while self.running:
            try:
                self._tick()
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(5)
        self.cleanup()

    def _tick(self):
        now = time.monotonic()

        # Check for button press on display server
        result = http_get("/buttons/check", timeout=2)
        if result.get("pressed"):
            print(f"[BUTTON] Pressed ({result.get('button')}) — forcing display update...")
            self._force_display_update()
            return

        # Time for a photo?
        if now - self.last_photo_time >= PHOTO_INTERVAL:
            self._photo_cycle(now)

        time.sleep(4)

    def _photo_cycle(self, now: float):
        self.last_photo_time = now

        print("[PHOTO] Capturing...")
        try:
            _, photo_uri = self.camera.capture()
        except Exception as e:
            print(f"[PHOTO] Camera error: {e}")
            return

        messages = self.ctx.build_request_data(photo_uri, CAMERA_WIDTH, CAMERA_HEIGHT)
        print(f"[CONTEXT] {len(self.ctx.messages)} msgs, ~{self.ctx.total_tokens()} tokens")
        obs = self.ctx.get_window_text()
        if obs:
            print(f"[OBSERVATIONS]\n{obs}\n---")
        for i, msg in enumerate(messages):
            c = msg.get("content", "")
            if isinstance(c, list):
                for part in c:
                    if part.get("type") == "text":
                        print(f"[PROMPT msg {i}]\n{part['text'][:2000]}")
                    elif part.get("type") == "image_url":
                        print(f"[PROMPT msg {i}] <image {CAMERA_WIDTH}x{CAMERA_HEIGHT}>")
            elif isinstance(c, str):
                print(f"[PROMPT msg {i}]\n{c[:500]}")
        print(f"[AI] Sending to {self.ai.model}...", flush=True)
        try:
            result = self.ai.reason_about_photo(messages)
        except Exception as e:
            print(f"[AI] API error: {e}")
            return

        if result.get("reasoning"):
            self.ctx.add_reasoning(result["reasoning"])
            print(f"[AI] Reasoning: {result['reasoning'][:120]}...")

        if result.get("should_display"):
            if not self._cooldown_passed():
                remaining = int(self.cooldown_until - now)
                print(f"[DISPLAY] AI wants to update but cooldown: {remaining}s remaining")
            else:
                self._do_display_update(result)

        self.ctx.check_compact(self.ai)

    def _force_display_update(self):
        try:
            _, photo_uri = self.camera.capture()
            observations = self.ctx.get_window_text()
            previous = self.ctx.get_previous_display()

            display_text = ""
            question = ""

            if observations.strip():
                result = self.ai.consolidate(observations, previous)
                display_text = result.get("display_text", "")
                question = result.get("question", "")
                print(f"[PARSE] display_text='{display_text[:50]}' question='{question}'")
            else:
                messages = self.ctx.build_request_data(photo_uri, CAMERA_WIDTH, CAMERA_HEIGHT)
                result = self.ai.reason_about_photo(messages)
                if result.get("reasoning"):
                    self.ctx.add_reasoning(result["reasoning"])
                display_text = result.get("display_text") or result.get("reasoning", "")[:200]
                question = result.get("question", "")

            if not display_text.strip():
                display_text = "..."

            self._send_to_display(display_text, question)

        except Exception as e:
            print(f"[FORCE] Error: {e}")

    def _do_display_update(self, result: dict):
        display_text = result.get("display_text", "").strip()
        question = result.get("question", "").strip()

        if not display_text:
            return

        self._send_to_display(display_text, question)

    def _send_to_display(self, display_text: str, question: str):
        print(f"[SEND] text='{display_text[:60]}'  question='{question}'")

        success = http_post("/display", {"text": display_text, "question": question}, timeout=10)
        if not success:
            print("[DISPLAY] Failed to send to display server")
            self._set_cooldown(DISPLAY_UPDATE_INTERVAL)
            return

        self.ctx.add_display(display_text)
        self.ctx.close_window(self.ai)

        if question:
            self._set_cooldown(DISPLAY_UPDATE_INTERVAL)
            print(f"[BUTTON] Waiting for response (timeout: {BUTTON_RESPONSE_TIMEOUT}s)...")
            result = http_get(f"/buttons/wait?timeout={BUTTON_RESPONSE_TIMEOUT}", timeout=BUTTON_RESPONSE_TIMEOUT + 10)
            answer = result.get("response")
            if answer:
                print(f"[BUTTON] User: {answer}")
                self.ctx.add_button_response(question, answer)
                self._set_cooldown(0)
            else:
                print("[BUTTON] Timed out")
        else:
            pass

    def cleanup(self):
        print("Cleaning up...")
        try:
            self.camera.close()
        except Exception:
            pass
        print("Done.")


def main():
    orch = Orchestrator()
    orch.run()


if __name__ == "__main__":
    main()
