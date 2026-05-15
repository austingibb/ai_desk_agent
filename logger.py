"""Timestamped logging — writes to journal (print) and optionally to a verbose log file."""
import os
import sys
from datetime import datetime

VERBOSE_LOG = os.environ.get("VERBOSE_LOG", "")


def info(msg: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    sys.stdout.flush()
    if VERBOSE_LOG:
        try:
            with open(VERBOSE_LOG, "a") as f:
                f.write(line + "\n")
        except Exception:
            pass
