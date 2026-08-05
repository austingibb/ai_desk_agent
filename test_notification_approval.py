import sys
import time
import types
import unittest
from unittest.mock import patch


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

import config
from main import Orchestrator


class _NotificationStore:
    def __init__(self):
        self.pending = {
            "id": "notif_test",
            "status": "proposed",
            "message": "Stretch every hour",
            "trigger_type": "interval",
            "trigger_value": "3600",
            "proposed_at": time.time(),
        }

    def get_pending_proposal(self):
        return self.pending

    def approve_pending(self):
        return self._resolve("approved")

    def reject_pending(self):
        return self._resolve("rejected")

    def _resolve(self, status):
        if not self.pending:
            return None
        resolved = self.pending
        resolved["status"] = status
        self.pending = None
        return resolved


class SmartNotificationApprovalTests(unittest.TestCase):
    def _orchestrator(self):
        orch = object.__new__(Orchestrator)
        orch.notification_store = _NotificationStore()
        orch.last_chat_message_time = 0
        return orch

    def test_smart_tool_approves_after_real_chat_response(self):
        orch = self._orchestrator()
        orch.last_chat_message_time = time.time() + 1

        with patch("main.NOTIFICATION_APPROVAL_MODE", "smart"):
            result = orch._tool_resolve_notification_proposal({
                "decision": "approve",
                "reason": "The user said it would help while studying.",
            })

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["decision"], "approve")
        self.assertIsNone(orch.notification_store.pending)

    def test_smart_tool_cannot_approve_before_user_response(self):
        orch = self._orchestrator()

        with patch("main.NOTIFICATION_APPROVAL_MODE", "smart"):
            result = orch._tool_resolve_notification_proposal({
                "decision": "approve",
                "reason": "No user response yet.",
            })

        self.assertEqual(result["status"], "error")
        self.assertIsNotNone(orch.notification_store.pending)

    def test_smart_tool_rejects_after_agent_interprets_user_response(self):
        orch = self._orchestrator()
        orch.last_chat_message_time = time.time() + 1

        with patch("main.NOTIFICATION_APPROVAL_MODE", "smart"):
            result = orch._tool_resolve_notification_proposal({
                "decision": "reject",
                "reason": "The user said reminders would interrupt their focus.",
            })

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["decision"], "reject")
        self.assertIsNone(orch.notification_store.pending)

    def test_resolver_tool_is_only_exposed_in_smart_mode(self):
        with patch("config.NOTIFICATION_APPROVAL_MODE", "smart"):
            smart_names = {
                tool["function"]["name"] for tool in config.get_tool_definitions()
            }
        with patch("config.NOTIFICATION_APPROVAL_MODE", "legacy"):
            legacy_names = {
                tool["function"]["name"] for tool in config.get_tool_definitions()
            }

        self.assertIn("resolve_notification_proposal", smart_names)
        self.assertNotIn("resolve_notification_proposal", legacy_names)


if __name__ == "__main__":
    unittest.main()
