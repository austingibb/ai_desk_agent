"""Local-only version history for vision requests and description logging."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import threading
from datetime import datetime, timezone

from config import (
    VISION_DESCRIPTION_LOG_FILE,
    VISION_REQUESTS_LEGACY_FILE,
    VISION_REQUESTS_REPO_DIR,
)


DEFAULT_REQUESTS = "# Requests for Image Model\n"
REQUESTS_HEADING = "# Requests for Image Model"


class VisionRequestHistory:
    """A nested, local Git repository containing only the requests file."""

    def __init__(
        self,
        repo_dir: str = VISION_REQUESTS_REPO_DIR,
        legacy_file: str | None = VISION_REQUESTS_LEGACY_FILE,
    ):
        self.repo_dir = os.path.abspath(repo_dir)
        self.file_path = os.path.join(self.repo_dir, "requests_for_image_model.md")
        self.legacy_file = os.path.abspath(legacy_file) if legacy_file else None
        self._lock = threading.RLock()
        self._ensure_repo()

    def _git(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        result = subprocess.run(
            ["git", *args],
            cwd=self.repo_dir,
            text=True,
            capture_output=True,
        )
        if check and result.returncode != 0:
            detail = (result.stderr or result.stdout).strip()
            raise RuntimeError(f"vision request Git command failed: {detail}")
        return result

    def _ensure_repo(self):
        with self._lock:
            os.makedirs(self.repo_dir, exist_ok=True)
            if not os.path.isdir(os.path.join(self.repo_dir, ".git")):
                self._git("init", "--quiet")
                self._git("config", "user.name", "Vision Request History")
                self._git("config", "user.email", "vision-history@localhost")

            migrated_legacy = False
            if not os.path.exists(self.file_path):
                if self.legacy_file and os.path.isfile(self.legacy_file):
                    shutil.copy2(self.legacy_file, self.file_path)
                    migrated_legacy = True
                else:
                    self._atomic_write(DEFAULT_REQUESTS)

            self.commit_if_changed("Initialize vision requests")
            if migrated_legacy:
                # The committed nested copy is now canonical and recoverable.
                os.remove(self.legacy_file)

    def _atomic_write(self, content: str):
        descriptor, temporary = tempfile.mkstemp(
            prefix=".vision-requests-",
            dir=self.repo_dir,
            text=True,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(content)
            os.replace(temporary, self.file_path)
        finally:
            if os.path.exists(temporary):
                os.remove(temporary)

    def read(self) -> str:
        with self._lock:
            try:
                with open(self.file_path, encoding="utf-8") as handle:
                    return handle.read()
            except FileNotFoundError:
                return ""

    def update(self, requests_text: str) -> str:
        """Replace the requests and create an independent local commit."""
        requests_text = requests_text.strip()
        while requests_text.startswith(REQUESTS_HEADING):
            requests_text = requests_text[len(REQUESTS_HEADING):].lstrip()
        if not requests_text:
            raise ValueError("vision requests cannot be empty")
        with self._lock:
            self._atomic_write(f"{REQUESTS_HEADING}\n\n{requests_text}\n")
            return self.commit_if_changed("Update vision requests")

    def snapshot(self) -> tuple[str, str]:
        """Return an immutable prompt snapshot and the commit that contains it."""
        with self._lock:
            commit = self.commit_if_changed()
            return self.read(), commit

    def commit_if_changed(self, message: str = "Record vision request changes") -> str:
        """Commit manual or programmatic edits and return the full HEAD hash."""
        with self._lock:
            self._git("add", "--", os.path.basename(self.file_path))
            diff = self._git(
                "diff",
                "--cached",
                "--quiet",
                "--",
                os.path.basename(self.file_path),
                check=False,
            )
            if diff.returncode == 1:
                self._git(
                    "commit",
                    "--quiet",
                    "-m",
                    message,
                    "--",
                    os.path.basename(self.file_path),
                )
            elif diff.returncode != 0:
                detail = (diff.stderr or diff.stdout).strip()
                raise RuntimeError(f"could not inspect vision request changes: {detail}")
            return self.current_commit()

    def current_commit(self) -> str:
        with self._lock:
            result = self._git("rev-parse", "HEAD")
            return result.stdout.strip()


class VisionDescriptionLog:
    """Append-only JSONL containing successful vision descriptions only."""

    def __init__(self, path: str = VISION_DESCRIPTION_LOG_FILE):
        self.path = os.path.abspath(path)
        self._lock = threading.Lock()

    def append(
        self,
        *,
        description: str,
        request_commit: str,
        model: str,
        provider: str,
        source: str,
        captured_at: float | None,
        latency_s: float,
        usage=None,
        timings=None,
    ) -> dict:
        record = {
            "captured_at": (
                datetime.fromtimestamp(captured_at, timezone.utc).isoformat()
                if captured_at is not None
                else None
            ),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "provider": provider,
            "model": model,
            "request_commit": request_commit,
            "latency_s": round(latency_s, 3),
            "description": description,
        }
        if usage is not None:
            record["usage"] = usage
        if timings is not None:
            record["timings"] = timings

        with self._lock:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record
