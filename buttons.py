"""Button input handling via gpiod for YES/NO responses."""

import time
import gpiod
from config import PIN_YES, PIN_NO, BUTTON_RESPONSE_TIMEOUT


class Buttons:
    def __init__(self, chip_path="/dev/gpiochip4"):
        self._pressed_since_last_check = False

        cfg = {}
        for pin in (PIN_YES, PIN_NO):
            cfg[pin] = gpiod.LineSettings(
                direction=gpiod.line.Direction.INPUT,
                bias=gpiod.line.Bias.PULL_UP,
                edge_detection=gpiod.line.Edge.FALLING,
            )
        self.request = gpiod.request_lines(chip_path, cfg, consumer="ai-eink-buttons")

    def wait_for_response(self, timeout: float = BUTTON_RESPONSE_TIMEOUT) -> str:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            remaining = max(1, int((deadline - time.monotonic()) * 1000))
            events = self.request.wait_edge_events(remaining)
            if events is False or events is None:
                continue
            for event in events:
                if event.event_type != gpiod.EdgeEvent.Type.FALLING_EDGE:
                    continue
                if event.line_offset == PIN_YES:
                    return "YES"
                elif event.line_offset == PIN_NO:
                    return "NO"
        return None

    def either_pressed(self) -> bool:
        events = self.request.wait_edge_events(1)
        if not events:
            return False
        for event in events:
            if event.event_type == gpiod.EdgeEvent.Type.FALLING_EDGE:
                return True
        return False

    def close(self):
        self.request.release()
