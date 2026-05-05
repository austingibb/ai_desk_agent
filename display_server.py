#!/usr/bin/env python3
"""HTTP display + button server for Pi Zero 2W (192.168.0.38:5050)."""

import json
import time
import signal
import sys
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from config import BUTTON_RESPONSE_TIMEOUT, PIN_YES, PIN_NO

import gpiod
from display import Display


DISPLAY_LOCK = threading.Lock()
display = None
button_request = None


class DisplayHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print(f"[HTTP] {args[0] if args else fmt}")

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json({"status": "ok", "display": display is not None})
        elif self.path.startswith("/buttons/check"):
            if not button_request.get_value(PIN_YES):
                self._send_json({"pressed": True, "button": "YES"})
            elif not button_request.get_value(PIN_NO):
                self._send_json({"pressed": True, "button": "NO"})
            else:
                self._send_json({"pressed": False})
        elif self.path.startswith("/buttons/wait"):
            timeout = BUTTON_RESPONSE_TIMEOUT
            if "?timeout=" in self.path:
                try:
                    timeout = float(self.path.split("timeout=")[1].split("&")[0])
                except ValueError:
                    pass
            self._wait_and_respond(timeout)
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path == "/display":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                data = json.loads(body)
            except json.JSONDecodeError:
                self._send_json({"error": "invalid json"}, 400)
                return

            text = data.get("text", "")
            question = data.get("question", "")

            with DISPLAY_LOCK:
                try:
                    display.show_text(text, question=question if question else None)
                    self._send_json({"status": "ok"})
                except Exception as e:
                    self._send_json({"error": str(e)}, 500)
        elif self.path == "/shutdown":
            self._send_json({"status": "shutting down"})
            threading.Thread(target=self.server.shutdown).start()
        else:
            self._send_json({"error": "not found"}, 404)

    def _wait_and_respond(self, timeout: float):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not button_request.get_value(PIN_YES):
                self._send_json({"response": "YES"})
                return
            if not button_request.get_value(PIN_NO):
                self._send_json({"response": "NO"})
                return
            time.sleep(0.1)
        self._send_json({"response": None})


def init_buttons():
    global yes_line, no_line, button_request
    cfg = {}
    for pin in (PIN_YES, PIN_NO):
        cfg[pin] = gpiod.LineSettings(direction=gpiod.line.Direction.INPUT, bias=gpiod.line.Bias.PULL_UP)
    button_request = gpiod.request_lines("/dev/gpiochip0", cfg, consumer="ai-eink-buttons")
    print("Buttons initialized.")


def main():
    global display

    print("Initializing display...")
    display = Display()
    display.show_booting()
    print("Display initialized.")

    init_buttons()

    server = HTTPServer(("0.0.0.0", 5050), DisplayHandler)
    print("Display server listening on :5050")

    def shutdown(sig, frame):
        print("\nShutting down...")
        server.shutdown()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        server.serve_forever()
    finally:
        print("Cleaning up...")
        if button_request:
            button_request.release()
        if display:
            display.show_text("Offline")
        print("Done.")


if __name__ == "__main__":
    main()
