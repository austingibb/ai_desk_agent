import base64
import http.client
import json
import os
import socket
import sys
import tempfile
import threading
import time
import types
import unittest
from http.server import ThreadingHTTPServer
from unittest.mock import patch


# ChatHandler itself is stdlib-only, but main.py also defines hardware/network
# orchestration. Stub optional imports so this route test runs before project
# dependencies are installed.
try:
    import dotenv  # noqa: F401
except ModuleNotFoundError:
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = dotenv_stub

try:
    import requests  # noqa: F401
except ModuleNotFoundError:
    requests_stub = types.ModuleType("requests")
    requests_stub.Timeout = type("Timeout", (Exception,), {})
    requests_stub.ConnectionError = type("ConnectionError", (Exception,), {})
    requests_stub.get = lambda *args, **kwargs: None
    requests_stub.post = lambda *args, **kwargs: None
    sys.modules["requests"] = requests_stub

try:
    import PIL  # noqa: F401
except ModuleNotFoundError:
    pil_stub = types.ModuleType("PIL")
    pil_stub.Image = object()
    sys.modules["PIL"] = pil_stub

from context import Context
from main import ChatHandler, ChatUIState, Orchestrator, _chat_asset_version


class _Notifications:
    def has_pending_proposal(self):
        return False


class _PendingNotifications:
    def __init__(self):
        self.pending = {
            "id": "notif_test",
            "message": "Stretch every hour",
            "status": "proposed",
        }
        self.approvals = 0
        self.rejections = 0

    def has_pending_proposal(self):
        return self.pending is not None

    def approve_pending(self):
        if not self.pending:
            return None
        self.approvals += 1
        resolved = self.pending
        self.pending = None
        return resolved

    def reject_pending(self):
        if not self.pending:
            return None
        self.rejections += 1
        resolved = self.pending
        self.pending = None
        return resolved


class _Presence:
    def touch(self):
        pass


class _Orchestrator:
    _chat_time = staticmethod(Orchestrator._chat_time)
    _queue_bubble = Orchestrator._queue_bubble
    _snapshot_chat_sources = Orchestrator._snapshot_chat_sources
    _find_chat_media = Orchestrator._find_chat_media
    _sync_chat_event_locked = Orchestrator._sync_chat_event_locked
    _undo_queued_message = Orchestrator._undo_queued_message
    _edit_queued_message = Orchestrator._edit_queued_message

    def __init__(self):
        self.ctx = Context()
        self.ctx_lock = threading.Lock()
        self.chat_queue = []
        self.chat_queue_lock = threading.Lock()
        self.chat_transfer_lock = threading.Lock()
        self.chat_event = threading.Event()
        self.ui_state = ChatUIState()
        self.sse_slots = threading.BoundedSemaphore(8)
        self.running = True
        self.notification_store = _Notifications()
        self.presence = _Presence()
        self.last_chat_message_time = 0.0


class ChatHTTPTests(unittest.TestCase):
    def setUp(self):
        # Most route tests exercise the optional upload path. Individual tests
        # turn this off to cover text-only models such as the default DeepSeek.
        self.image_support_patch = patch("main.LLM_SUPPORTS_IMAGES", True)
        self.image_support_patch.start()
        self.addCleanup(self.image_support_patch.stop)
        self.orch = _Orchestrator()
        ChatHandler.orchestrator = self.orch
        ChatHandler.session_token = "test-session"
        ChatHandler.use_https = False
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), ChatHandler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.conn = http.client.HTTPConnection(
            "127.0.0.1", self.server.server_port, timeout=5
        )

    def tearDown(self):
        self.orch.running = False
        with self.orch.ui_state.condition:
            self.orch.ui_state.condition.notify_all()
        self.conn.close()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)

    def _request(self, method, path, body=None):
        headers = {"Cookie": "session=test-session"}
        if body is not None:
            body = json.dumps(body)
            headers["Content-Type"] = "application/json"
        self.conn.request(method, path, body=body, headers=headers)
        return self.conn.getresponse()

    def test_upload_poll_and_authenticated_media_route(self):
        raw = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
        encoded = base64.b64encode(raw).decode()
        response = self._request(
            "POST",
            "/chat",
            {
                "message": "Look at this.",
                "images": [{
                    "name": "reaction.png",
                    "type": "image/png",
                    "data_url": f"data:image/png;base64,{encoded}",
                }],
            },
        )
        self.assertEqual(response.status, 200)
        posted_response = json.loads(response.read())
        self.assertTrue(posted_response["id"].startswith("q_"))
        self.assertTrue(posted_response["message"]["queued"])

        response = self._request("GET", "/chat")
        self.assertEqual(response.status, 200)
        payload = json.loads(response.read())
        posted = payload["messages"][0]
        self.assertEqual(posted["id"], posted_response["id"])
        self.assertEqual(posted["content"], "Look at this.")
        self.assertNotIn("base64", json.dumps(posted))
        media_url = posted["images"][0]["url"]

        response = self._request("GET", media_url)
        self.assertEqual(response.status, 200)
        self.assertEqual(response.getheader("Content-Type"), "image/png")
        self.assertEqual(response.read(), raw)

    def test_gif_upload_is_rejected(self):
        raw = b"GIF89a" + b"\x00" * 20
        response = self._request(
            "POST",
            "/chat",
            {
                "message": "Look at this.",
                "images": [{
                    "name": "reaction.gif",
                    "type": "image/gif",
                    "data_url": f"data:image/gif;base64,{base64.b64encode(raw).decode()}",
                }],
            },
        )
        self.assertEqual(response.status, 400)
        self.assertIn("static PNG, JPEG, or WebP", json.loads(response.read())["error"])
        self.assertEqual(self.orch.chat_queue, [])

    def test_homepage_query_and_rendered_javascript(self):
        response = self._request("GET", "/?")
        self.assertEqual(response.status, 200)
        html = response.read().decode()
        self.assertIn("/static/chat.mjs?v=", html)
        self.assertIn('type="importmap"', html)
        self.assertIn('accept="image/png,image/jpeg,image/webp"', html)
        self.assertIn("Attach static images (PNG, JPEG, or WebP)", html)
        self.assertIn('"supportsImages":true', html)
        self.assertNotIn("image/gif", html)
        self.assertNotIn("__CHAT_ASSET_VERSION__", html)
        self.assertNotIn("__LLM_SUPPORTS_IMAGES__", html)

    def test_text_only_model_rejects_images_and_disables_them_in_ui_config(self):
        raw = b"\x89PNG\r\n\x1a\n" + b"\x00" * 20
        encoded = base64.b64encode(raw).decode()
        with patch("main.LLM_SUPPORTS_IMAGES", False):
            response = self._request(
                "POST",
                "/chat",
                {
                    "message": "Look at this.",
                    "images": [{
                        "name": "reaction.png",
                        "type": "image/png",
                        "data_url": f"data:image/png;base64,{encoded}",
                    }],
                },
            )
            self.assertEqual(response.status, 400)
            self.assertIn("does not support image uploads", json.loads(response.read())["error"])
            self.assertEqual(self.orch.chat_queue, [])

            response = self._request("GET", "/")
            self.assertEqual(response.status, 200)
            self.assertIn('"supportsImages":false', response.read().decode())

            response = self._request("POST", "/chat", {"message": "Text still works."})
            self.assertEqual(response.status, 200)
            response.read()
            self.assertEqual(self.orch.chat_queue[0]["content"], "Text still works.")

    def test_smart_approval_leaves_natural_language_decision_for_agent(self):
        notifications = _PendingNotifications()
        self.orch.notification_store = notifications

        with patch("main.NOTIFICATION_APPROVAL_MODE", "smart"):
            response = self._request(
                "POST",
                "/chat",
                {"message": "That sounds useful while I'm studying."},
            )
            self.assertEqual(response.status, 200)
            response.read()

        self.assertEqual(notifications.approvals, 0)
        self.assertEqual(notifications.rejections, 0)
        self.assertEqual(len(self.orch.chat_queue), 1)
        self.assertEqual(self.orch.chat_queue[0]["text"], "That sounds useful while I'm studying.")
        self.assertGreater(self.orch.last_chat_message_time, 0)

    def test_legacy_mode_does_not_scan_chat_keywords(self):
        notifications = _PendingNotifications()
        self.orch.notification_store = notifications

        with (
            patch("main.NOTIFICATION_APPROVAL_MODE", "legacy"),
            patch("main.ENABLE_DISPLAY", False),
        ):
            response = self._request(
                "POST",
                "/chat",
                {"message": "I know nothing about that yet."},
            )
            self.assertEqual(response.status, 200)
            response.read()

        self.assertEqual(notifications.approvals, 0)
        self.assertEqual(notifications.rejections, 0)
        self.assertEqual(len(self.orch.chat_queue), 1)
        self.assertEqual(self.orch.chat_queue[0]["text"], "I know nothing about that yet.")

    def test_identical_posts_receive_distinct_stable_ids(self):
        ids = []
        for _ in range(2):
            response = self._request("POST", "/chat", {"message": "same"})
            self.assertEqual(response.status, 200)
            ids.append(json.loads(response.read())["id"])
        self.assertNotEqual(ids[0], ids[1])

        response = self._request("GET", "/chat")
        payload = json.loads(response.read())
        self.assertEqual([message["id"] for message in payload["messages"]], ids)
        self.assertTrue(all(message["queued"] for message in payload["messages"]))
        first_snapshot = payload["messages"]

        response = self._request("GET", "/chat")
        repeated = json.loads(response.read())["messages"]
        self.assertEqual(repeated, first_snapshot)

    def test_edit_preserves_queue_position_id_and_attachments(self):
        raw = b"\x89PNG\r\n\x1a\n" + b"\x01" * 20
        response = self._request(
            "POST", "/chat", {
                "message": "typo",
                "images": [{
                    "name": "edit.png",
                    "type": "image/png",
                    "data_url": f"data:image/png;base64,{base64.b64encode(raw).decode()}",
                }],
            },
        )
        queue_id = json.loads(response.read())["id"]
        second = self._request("POST", "/chat", {"message": "second"})
        second_id = json.loads(second.read())["id"]

        response = self._request(
            "PATCH", f"/chat/queue/{queue_id}", {"message": "fixed"}
        )
        self.assertEqual(response.status, 200)
        updated = json.loads(response.read())["message"]
        self.assertEqual(updated["id"], queue_id)
        self.assertEqual(updated["content"], "fixed")
        self.assertEqual(len(updated["images"]), 1)
        self.assertEqual([entry["id"] for entry in self.orch.chat_queue], [queue_id, second_id])

    def test_empty_edit_guidance_and_conflict_after_sweep(self):
        response = self._request("POST", "/chat", {"message": "remove me"})
        queue_id = json.loads(response.read())["id"]

        response = self._request("PATCH", f"/chat/queue/{queue_id}", {"message": ""})
        self.assertEqual(response.status, 400)
        self.assertIn("Use Undo send instead", json.loads(response.read())["error"])

        Orchestrator._sweep_chat_queue(self.orch)
        response = self._request("DELETE", f"/chat/queue/{queue_id}")
        self.assertEqual(response.status, 409)
        response.read()
        response = self._request("PATCH", f"/chat/queue/{queue_id}", {"message": "late"})
        self.assertEqual(response.status, 409)
        response.read()

    def test_undo_returns_media_in_one_response(self):
        raw = b"\x89PNG\r\n\x1a\n" + b"\x02" * 20
        data_url = f"data:image/png;base64,{base64.b64encode(raw).decode()}"
        response = self._request(
            "POST", "/chat", {
                "message": "restore",
                "images": [{"name": "undo.png", "type": "image/png", "data_url": data_url}],
            },
        )
        queue_id = json.loads(response.read())["id"]

        response = self._request("DELETE", f"/chat/queue/{queue_id}")
        self.assertEqual(response.status, 200)
        restored = json.loads(response.read())["restored"]
        self.assertEqual(restored["text"], "restore")
        self.assertEqual(restored["images"][0]["data_url"], data_url)
        self.assertEqual(self.orch.chat_queue, [])

    def test_context_bubbles_always_precede_queue_bubbles(self):
        self.orch.ctx.add_user("sent")
        self.orch.ctx.add_assistant({
            "content": "",
            "tool_calls": [
                {"id": "one", "name": "update_display", "arguments": {"text": "first"}},
                {"id": "two", "name": "send_chat_message", "arguments": {"text": "second"}},
            ],
        })
        self.orch.chat_queue.append({
            "id": "q_" + "1" * 32,
            "created_at": 1.0,
            "text": "older queued timestamp",
            "content": "older queued timestamp",
            "chat_images": [],
        })

        response = self._request("GET", "/chat")
        payload = json.loads(response.read())
        messages = payload["messages"]
        self.assertEqual([message["queued"] for message in messages], [False, False, False, True])
        assistant_ids = [message["id"] for message in messages if message["role"] == "assistant"]
        self.assertEqual(len(assistant_ids), len(set(assistant_ids)))
        self.assertTrue(assistant_ids[0].endswith(":assistant:0"))
        self.assertTrue(assistant_ids[1].endswith(":assistant:1"))
        self.assertIn("agent", payload)
        self.assertIn("chat_revision", payload)

    def test_static_assets_are_authenticated_and_versioned(self):
        unauthenticated = http.client.HTTPConnection(
            "127.0.0.1", self.server.server_port, timeout=5
        )
        unauthenticated.request("GET", "/static/chat.mjs")
        response = unauthenticated.getresponse()
        self.assertEqual(response.status, 401)
        response.read()
        unauthenticated.close()

        response = self._request("GET", "/static/chat.mjs?v=test")
        self.assertEqual(response.status, 200)
        self.assertIn("immutable", response.getheader("Cache-Control"))
        self.assertEqual(response.getheader("ETag"), '"dev"')
        response.read()

    def test_static_asset_version_changes_with_asset_contents(self):
        with tempfile.TemporaryDirectory() as static_dir:
            for filename in ("chat.css", "chat.mjs", "chat_model.mjs"):
                with open(os.path.join(static_dir, filename), "wb") as asset:
                    asset.write(filename.encode())
            first = _chat_asset_version(static_dir)
            with open(os.path.join(static_dir, "chat.mjs"), "ab") as asset:
                asset.write(b" changed")
            second = _chat_asset_version(static_dir)

        self.assertEqual(len(first), 12)
        self.assertNotEqual(first, second)

    def test_event_stream_cap_returns_503(self):
        self.orch.sse_slots = threading.BoundedSemaphore(1)
        stream = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        stream.request("GET", "/chat/events", headers={"Cookie": "session=test-session"})
        first = stream.getresponse()
        self.assertEqual(first.status, 200)

        overflow = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        overflow.request("GET", "/chat/events", headers={"Cookie": "session=test-session"})
        second = overflow.getresponse()
        self.assertEqual(second.status, 503)
        second.read()
        overflow.close()

        first.fp.raw._sock.shutdown(socket.SHUT_RDWR)
        first.close()
        stream.close()
        released = False
        for _ in range(20):
            self.orch.ui_state.chat_changed()
            if self.orch.sse_slots.acquire(blocking=False):
                self.orch.sse_slots.release()
                released = True
                break
            time.sleep(0.05)
        self.assertTrue(released)

        replacement = http.client.HTTPConnection(
            "127.0.0.1", self.server.server_port, timeout=5
        )
        replacement.request(
            "GET", "/chat/events", headers={"Cookie": "session=test-session"}
        )
        third = replacement.getresponse()
        self.assertEqual(third.status, 200)
        replacement.close()
        self.orch.ui_state.chat_changed()

    def test_event_stream_emits_state_and_transcript_invalidations(self):
        stream = http.client.HTTPConnection(
            "127.0.0.1", self.server.server_port, timeout=5
        )
        stream.request(
            "GET", "/chat/events", headers={"Cookie": "session=test-session"}
        )
        response = stream.getresponse()
        self.assertEqual(response.status, 200)

        def read_event():
            lines = []
            while True:
                line = response.fp.readline().decode()
                if not line or line == "\n":
                    return "".join(lines)
                lines.append(line)

        self.assertEqual(read_event(), "retry: 2000\n")
        initial = read_event()
        self.assertIn("event: snapshot", initial)
        self.assertNotIn("id:", initial)

        self.orch.ui_state.set_agent("acting", "testing events")
        state_event = read_event()
        self.assertIn('"mode":"acting"', state_event)
        initial_chat_revision = self.orch.ui_state.snapshot()["chat_revision"]

        self.orch.ui_state.chat_changed()
        chat_event = read_event()
        self.assertIn(
            f'"chat_revision":{initial_chat_revision + 1}', chat_event
        )
        self.assertNotIn("id:", state_event + chat_event)
        stream.close()
        self.orch.ui_state.chat_changed()

    def test_event_stream_closes_after_idle_timeout(self):
        with (
            patch("main.CHAT_SSE_IDLE_SECONDS", 0.05),
            patch("main.CHAT_SSE_HEARTBEAT_SECONDS", 0.01),
        ):
            stream = http.client.HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
            stream.request("GET", "/chat/events", headers={"Cookie": "session=test-session"})
            response = stream.getresponse()
            self.assertEqual(response.status, 200)
            body = response.read().decode()
            self.assertIn("event: snapshot", body)
            self.assertNotIn("id:", body)
            stream.close()


if __name__ == "__main__":
    unittest.main()
