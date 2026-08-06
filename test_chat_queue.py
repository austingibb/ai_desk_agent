import threading
import time
import types
import unittest
import sys
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

from context import Context
from main import ChatUIState, Orchestrator


def queue_entry(queue_id, text, created_at=1.0):
    return {
        "id": queue_id,
        "created_at": created_at,
        "text": text,
        "content": text,
        "chat_images": [],
    }


class ChatQueueTests(unittest.TestCase):
    def make_orchestrator(self):
        orch = Orchestrator.__new__(Orchestrator)
        orch.ctx = Context()
        orch.ctx_lock = threading.Lock()
        orch.chat_queue = []
        orch.chat_queue_lock = threading.Lock()
        orch.chat_transfer_lock = threading.Lock()
        orch.chat_event = threading.Event()
        orch.motion_event = threading.Event()
        orch.ui_state = ChatUIState()
        orch.running = True
        orch.pomodoro_mode = False
        return orch

    def test_sweep_merges_queue_with_one_context_call(self):
        orch = self.make_orchestrator()
        orch.chat_queue.extend([
            queue_entry("q_1", "first"),
            queue_entry("q_2", "correction", 2.0),
        ])
        orch.chat_event.set()

        context_id = orch._sweep_chat_queue()

        self.assertTrue(context_id.startswith("c_"))
        self.assertEqual(orch.chat_queue, [])
        self.assertFalse(orch.chat_event.is_set())
        self.assertEqual(len(orch.ctx.messages), 1)
        self.assertEqual(orch.ctx.messages[0]["content"], "first\ncorrection")
        self.assertEqual(orch.ctx.messages[0]["_chat_original_text"], "first\ncorrection")

    def test_wait_batch_executes_every_call_before_sweeping(self):
        orch = self.make_orchestrator()
        orch.chat_queue.append(queue_entry("q_1", "queued"))
        orch.chat_event.set()
        calls = []

        def execute_tool(_self, name, arguments):
            calls.append(name)
            return {"status": "ok", "name": name}

        orch._execute_tool = types.MethodType(execute_tool, orch)
        batch = [
            {"id": "call_wait", "name": "wait", "arguments": {"seconds": 10}},
            {"id": "call_schedule", "name": "schedule_notification", "arguments": {"id": "n1"}},
        ]
        orch.ctx.add_assistant({"content": "", "tool_calls": batch})

        orch._execute_tool_batch(batch)

        self.assertEqual(calls, ["wait", "schedule_notification"])
        self.assertEqual(
            [message["role"] for message in orch.ctx.messages],
            ["assistant", "tool", "tool", "user"],
        )
        self.assertEqual(orch.ctx.messages[-1]["content"], "queued")

    def test_batch_without_wait_leaves_queue_unsent(self):
        orch = self.make_orchestrator()
        orch.chat_queue.append(queue_entry("q_1", "hold"))
        orch.chat_event.set()
        orch._execute_tool = types.MethodType(
            lambda _self, name, arguments: {"status": "ok"}, orch
        )

        orch._execute_tool_batch([
            {"id": "call_action", "name": "log_drink", "arguments": {}}
        ])

        self.assertEqual([entry["id"] for entry in orch.chat_queue], ["q_1"])
        self.assertEqual([message["role"] for message in orch.ctx.messages], ["tool"])

    def test_tool_state_uses_raw_fallback_and_long_marklist(self):
        orch = self.make_orchestrator()
        observed = []

        def dispatch(_self, name, arguments):
            observed.append(_self.ui_state.snapshot()["agent"])
            return {"status": "ok"}

        orch._dispatch_tool = types.MethodType(dispatch, orch)
        Orchestrator._execute_tool(orch, "unmapped_tool", {})
        Orchestrator._execute_tool(orch, "capture_photo", {})

        self.assertEqual(observed[0]["mode"], "acting")
        self.assertEqual(observed[0]["detail"], "unmapped_tool")
        self.assertEqual(observed[1]["mode"], "acting_long")
        self.assertEqual(observed[1]["detail"], "capturing a fresh room photo")
        self.assertEqual(orch.ui_state.snapshot()["agent"]["mode"], "thinking")

    def test_blocked_state_tracks_lock_independently(self):
        orch = self.make_orchestrator()
        with orch._blocked_agent_state("compacting memory", True):
            compacting = orch.ui_state.snapshot()["agent"]
        with orch._blocked_agent_state("LLM backoff after failures", False):
            backoff = orch.ui_state.snapshot()["agent"]

        self.assertTrue(compacting["locks_input"])
        self.assertFalse(backoff["locks_input"])

    def test_idle_entry_is_a_sweep_boundary(self):
        orch = self.make_orchestrator()
        orch.chat_queue.append(queue_entry("q_1", "already waiting"))
        orch.chat_event.set()
        orch._idle_wait_impl = types.MethodType(lambda _self: None, orch)

        Orchestrator._idle_wait(orch)

        self.assertEqual(orch.chat_queue, [])
        self.assertEqual(orch.ctx.messages[-1]["content"], "already waiting")

    def test_chat_event_wakes_idle_wait_and_sweeps_immediately(self):
        orch = self.make_orchestrator()
        with patch("main.ENABLE_DISPLAY", False), patch("main.IDLE_TIMEOUT", 5):
            waiter = threading.Thread(target=orch._idle_wait_impl)
            waiter.start()
            time.sleep(0.02)
            with orch.chat_queue_lock:
                orch.chat_queue.append(queue_entry("q_1", "wake now"))
                orch.chat_event.set()
            waiter.join(timeout=0.5)

        self.assertFalse(waiter.is_alive())
        self.assertEqual(orch.chat_queue, [])
        self.assertEqual(orch.ctx.messages[-1]["content"], "wake now")

    def test_transfer_snapshot_observes_each_boundary_message_once(self):
        orch = self.make_orchestrator()
        orch.chat_queue.append(queue_entry("q_1", "at boundary"))
        orch.chat_event.set()
        preparing = threading.Event()
        release = threading.Event()
        snapshot_started = threading.Event()
        snapshot = {}

        from main import merge_queued_messages as real_merge

        def paused_merge(entries):
            preparing.set()
            self.assertTrue(release.wait(timeout=1))
            return real_merge(entries)

        def take_snapshot():
            snapshot_started.set()
            snapshot["value"] = orch._snapshot_chat_sources()

        with patch("main.merge_queued_messages", side_effect=paused_merge):
            sweep = threading.Thread(target=orch._sweep_chat_queue)
            sweep.start()
            self.assertTrue(preparing.wait(timeout=1))

            # POSTs do not take the transfer gate, so this is a deliberate tail
            # record created after the sweep boundary.
            with orch.chat_queue_lock:
                orch.chat_queue.append(queue_entry("q_2", "new tail", 2.0))
                orch.chat_event.set()

            reader = threading.Thread(target=take_snapshot)
            reader.start()
            self.assertTrue(snapshot_started.wait(timeout=1))
            time.sleep(0.02)
            self.assertTrue(reader.is_alive())
            release.set()
            sweep.join(timeout=1)
            reader.join(timeout=1)

        self.assertFalse(sweep.is_alive())
        self.assertFalse(reader.is_alive())
        context_messages, queue_messages = snapshot["value"]
        self.assertEqual(
            [message["content"] for message in context_messages], ["at boundary"]
        )
        self.assertEqual([message["text"] for message in queue_messages], ["new tail"])
        self.assertTrue(orch.chat_event.is_set())


if __name__ == "__main__":
    unittest.main()
