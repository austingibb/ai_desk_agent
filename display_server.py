#!/usr/bin/env python3
"""HTTP display + button server for Pi Zero 2W (192.168.0.38:5050)."""

import json
import time
import signal
import sys
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from config import PIN_YES, PIN_NO

import gpiod
from display import Display
from logger import info


DISPLAY_LOCK = threading.Lock()
display = None
button_request = None
button_state = {"button": None, "timestamp": None}
button_state_lock = threading.Lock()


def button_monitor_thread():
    global button_state
    while True:
        with button_state_lock:
            if button_state["button"] is None:
                if not button_request.get_value(PIN_YES):
                    button_state["button"] = "YES"
                    button_state["timestamp"] = time.time()
                    info(f"[BUTTON] YES pressed")
                elif not button_request.get_value(PIN_NO):
                    button_state["button"] = "NO"
                    button_state["timestamp"] = time.time()
                    info(f"[BUTTON] NO pressed")
        time.sleep(0.1)


class DisplayHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        info(f"[HTTP] {args[0] if args else fmt}")

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
        elif self.path == "/buttons/state":
            with button_state_lock:
                self._send_json({
                    "pressed": button_state["button"] is not None,
                    "button": button_state["button"],
                })
        elif self.path == "/buttons/check":
            if not button_request.get_value(PIN_YES):
                self._send_json({"pressed": True, "button": "YES"})
            elif not button_request.get_value(PIN_NO):
                self._send_json({"pressed": True, "button": "NO"})
            else:
                self._send_json({"pressed": False})
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

            with DISPLAY_LOCK:
                try:
                    display.show_text(text)
                    self._send_json({"status": "ok"})
                except Exception as e:
                    self._send_json({"error": str(e)}, 500)
        elif self.path == "/buttons/reset":
            with button_state_lock:
                button_state["button"] = None
                button_state["timestamp"] = None
            self._send_json({"status": "ok"})
        elif self.path == "/shutdown":
            self._send_json({"status": "shutting down"})
            threading.Thread(target=self.server.shutdown).start()
        else:
            self._send_json({"error": "not found"}, 404)


def init_buttons():
    global button_request
    cfg = {}
    for pin in (PIN_YES, PIN_NO):
        cfg[pin] = gpiod.LineSettings(direction=gpiod.line.Direction.INPUT, bias=gpiod.line.Bias.PULL_UP)
    button_request = gpiod.request_lines("/dev/gpiochip0", cfg, consumer="ai-eink-buttons")
    info("Buttons initialized.")


def main():
    global display

    info("Initializing display...")
    display = Display()
    display.show_booting()
    info("Display initialized.")

    init_buttons()

    t = threading.Thread(target=button_monitor_thread, daemon=True)
    t.start()

    server = HTTPServer(("0.0.0.0", 5050), DisplayHandler)
    info("Display server listening on :5050")

    def shutdown(sig, frame):
        info("\nShutting down...")
        server.shutdown()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        server.serve_forever()
    finally:
        info("Cleaning up...")
        if button_request:
            button_request.release()
        if display:
            display.show_text("Offline")
        info("Done.")


if __name__ == "__main__":
    main()
