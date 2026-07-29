import base64
import http.client
import json
import sys
import threading
import types
import unittest
from http.server import HTTPServer


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
from main import ChatHandler


class _Notifications:
    def has_pending_proposal(self):
        return False


class _Presence:
    def touch(self):
        pass


class _Orchestrator:
    def __init__(self):
        self.ctx = Context()
        self.ctx_lock = threading.Lock()
        self.chat_queue = []
        self.chat_queue_lock = threading.Lock()
        self.chat_event = threading.Event()
        self.status_lock = threading.Lock()
        self.status_message = ""
        self.notification_store = _Notifications()
        self.presence = _Presence()


class ChatHTTPTests(unittest.TestCase):
    def setUp(self):
        self.orch = _Orchestrator()
        ChatHandler.orchestrator = self.orch
        ChatHandler.session_token = "test-session"
        ChatHandler.use_https = False
        self.server = HTTPServer(("127.0.0.1", 0), ChatHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.conn = http.client.HTTPConnection(
            "127.0.0.1", self.server.server_port, timeout=5
        )

    def tearDown(self):
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
        raw = b"GIF89a" + b"\x00" * 20
        encoded = base64.b64encode(raw).decode()
        response = self._request(
            "POST",
            "/chat",
            {
                "message": "Look at this.",
                "images": [{
                    "name": "reaction.gif",
                    "type": "image/gif",
                    "data_url": f"data:image/gif;base64,{encoded}",
                }],
            },
        )
        self.assertEqual(response.status, 200)
        response.read()

        response = self._request("GET", "/chat")
        self.assertEqual(response.status, 200)
        payload = json.loads(response.read())
        posted = payload["messages"][0]
        self.assertEqual(posted["content"], "Look at this.")
        self.assertNotIn("base64", json.dumps(posted))
        media_url = posted["images"][0]["url"]

        response = self._request("GET", media_url)
        self.assertEqual(response.status, 200)
        self.assertEqual(response.getheader("Content-Type"), "image/gif")
        self.assertEqual(response.read(), raw)

    def test_homepage_query_and_rendered_javascript(self):
        response = self._request("GET", "/?")
        self.assertEqual(response.status, 200)
        html = response.read().decode()
        self.assertIn("replace(/\\n/g,'<br>')", html)
        self.assertNotIn("replace(/\n/g,'<br>')", html)


if __name__ == "__main__":
    unittest.main()
