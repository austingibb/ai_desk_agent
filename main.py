#!/usr/bin/env python3
"""AI E-Ink Roommate — agent loop orchestrator. Runs on Pi 5 (192.168.0.39)."""

import time
import signal
import sys
import json
import requests
from config import (
    DISPLAY_SERVER_URL,
    SYSTEM_PROMPT,
    TOOL_DEFINITIONS,
    MAX_TOOL_CALLS_PER_TURN,
    MIN_PHOTO_INTERVAL,
    MIN_DISPLAY_INTERVAL,
    MAX_WAIT_SECONDS,
    IDLE_TIMEOUT,
    BUTTON_CHECK_INTERVAL,
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
        self.camera = Camera()
        self.ai = AIClient()
        self.running = True
        self.last_photo_time = 0
        self.last_display_time = 0

        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame):
        print("\nShutting down...")
        self.running = False

    def run(self):
        print("Init camera...")
        print("Init AI client...")
        self.ctx.add_system(SYSTEM_PROMPT)
        self.ctx.add_user("You've just been powered on. Use take_photo to see the room for the first time.")
        print("Entering agent loop.")

        while self.running:
            try:
                self._turn()
            except Exception as e:
                print(f"[ERROR] {e}")
                time.sleep(5)

        self.cleanup()

    def _turn(self):
        tool_call_count = 0

        while self.running:
            tools = TOOL_DEFINITIONS if tool_call_count < MAX_TOOL_CALLS_PER_TURN else None
            if tools is None:
                print("[SAFETY] Tool call limit reached, forcing text-only response")

            print(f"[LLM] Sending {len(self.ctx.get_messages())} messages (~{self.ctx.total_tokens()} tokens)...")
            try:
                response = self.ai.chat_with_tools(self.ctx.get_messages(), tools)
            except Exception as e:
                print(f"[LLM] Error: {e}")
                time.sleep(5)
                return

            self.ctx.add_assistant(response)

            if response["reasoning"]:
                print(f"[REASONING] {response['reasoning'][:200]}...")
            if response["content"]:
                print(f"[AI] {response['content'][:200]}")

            if not response["tool_calls"]:
                print("[IDLE] AI produced no tool calls. Waiting...")
                self._idle_wait()
                return

            for tc in response["tool_calls"]:
                tool_call_count += 1
                print(f"[TOOL] {tc['name']}({tc['arguments']})")
                result = self._execute_tool(tc["name"], tc["arguments"])
                print(f"[TOOL RESULT] {json.dumps(result)[:200]}")
                self.ctx.add_tool_result(tc["id"], tc["name"], result)

            self.ctx.check_compact(self.ai)

    def _execute_tool(self, name: str, args: dict) -> dict:
        if name == "take_photo":
            return self._tool_take_photo()
        elif name == "update_display":
            return self._tool_update_display(args)
        elif name == "poll_buttons":
            return self._tool_poll_buttons()
        elif name == "wait":
            return self._tool_wait(args)
        else:
            return {"error": f"Unknown tool: {name}. Available: take_photo, update_display, poll_buttons, wait"}

    def _tool_take_photo(self) -> dict:
        elapsed = time.monotonic() - self.last_photo_time
        if elapsed < MIN_PHOTO_INTERVAL:
            time.sleep(MIN_PHOTO_INTERVAL - elapsed)

        try:
            _, photo_uri = self.camera.capture()
        except Exception as e:
            return {"status": "error", "message": f"Camera error: {e}"}

        self.last_photo_time = time.monotonic()
        self.ctx.add_image(photo_uri)
        return {"status": "ok", "description": "Photo captured and added to conversation."}

    def _tool_update_display(self, args: dict) -> dict:
        text = args.get("text", "")
        question = args.get("question", "")

        if not text.strip():
            return {"status": "error", "message": "No text provided"}

        elapsed = time.monotonic() - self.last_display_time
        if elapsed < MIN_DISPLAY_INTERVAL:
            time.sleep(MIN_DISPLAY_INTERVAL - elapsed)

        success = http_post("/display", {"text": text, "question": question}, timeout=15)
        self.last_display_time = time.monotonic()

        http_post("/buttons/reset", {}, timeout=5)

        if success:
            return {"status": "ok", "message": "Display updated."}
        else:
            return {"status": "error", "message": "Failed to communicate with display server"}

    def _tool_poll_buttons(self) -> dict:
        result = http_get("/buttons/state", timeout=5)
        button = result.get("button")
        return {"pressed": button is not None, "button": button}

    def _tool_wait(self, args: dict) -> dict:
        seconds = max(5, min(MAX_WAIT_SECONDS, args.get("seconds", 60)))
        print(f"[WAIT] Sleeping {seconds}s...")

        start = time.monotonic()
        while time.monotonic() - start < seconds:
            if not self.running:
                return {"status": "interrupted", "reason": "shutdown", "waited": int(time.monotonic() - start)}

            result = http_get("/buttons/state", timeout=2)
            if result.get("button"):
                waited = int(time.monotonic() - start)
                return {
                    "status": "interrupted",
                    "reason": f"Button {result['button']} pressed",
                    "button": result["button"],
                    "waited": waited,
                }
            time.sleep(BUTTON_CHECK_INTERVAL)

        return {"status": "ok", "waited": seconds}

    def _idle_wait(self):
        for _ in range(IDLE_TIMEOUT):
            if not self.running:
                return
            time.sleep(1)
        self.ctx.add_user(
            "Some time has passed. Use take_photo to see the room, or wait to stay quiet."
        )

    def cleanup(self):
        print("Cleaning up...")
        try:
            self.camera.close()
        except Exception:
            pass
        print("Done.")


def main():
    orch = Orchestrator()
    try:
        orch.run()
    finally:
        orch.cleanup()


if __name__ == "__main__":
    main()
