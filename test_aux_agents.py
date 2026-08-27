import json
import os
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import Mock, patch

try:
    import dotenv  # noqa: F401
except ModuleNotFoundError:
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = dotenv_stub

try:
    import requests
except ModuleNotFoundError:
    requests_stub = types.ModuleType("requests")
    requests_stub.Timeout = type("Timeout", (Exception,), {})
    requests_stub.ConnectionError = type("ConnectionError", (Exception,), {})
    requests_stub.get = lambda *args, **kwargs: None
    requests_stub.post = lambda *args, **kwargs: None
    sys.modules["requests"] = requests_stub
    requests = requests_stub

try:
    import PIL  # noqa: F401
except ModuleNotFoundError:
    pil_stub = types.ModuleType("PIL")
    pil_stub.Image = object()
    sys.modules["PIL"] = pil_stub

import aux_agents
from aux_agents import (
    AuxiliaryConfigError,
    AuxiliaryTurn,
    load_auxiliary_manager,
)
from context import Context
from main import Orchestrator


def tool_definition(name):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Use {name} mechanically.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    }


def classifier_response(delegate):
    return {
        "choices": [{
            "message": {
                "content": json.dumps({"delegate": delegate}),
            }
        }]
    }


def work_response(name=None, arguments=None, content="private prose"):
    tool_calls = []
    if name:
        tool_calls.append({
            "id": "provider_reused_id",
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(arguments or {}),
            },
        })
    return {
        "choices": [{
            "message": {
                "content": content,
                "tool_calls": tool_calls,
            }
        }]
    }


class FakeResponse:
    status_code = 200

    def __init__(self, data):
        self.data = data
        self.text = json.dumps(data)

    def raise_for_status(self):
        return None

    def json(self):
        return self.data


class QueuePost:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return FakeResponse(result)


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class AuxiliaryManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.config_path = os.path.join(self.temp_dir.name, "aux_agents.json")
        self.log_path = os.path.join(self.temp_dir.name, "aux_logs", "events.jsonl")

    def write_config(self, agents, schema_version=1):
        with open(self.config_path, "w", encoding="utf-8") as handle:
            json.dump({"schema_version": schema_version, "agents": agents}, handle)

    def load(
        self,
        agents,
        *,
        active=("log_drink", "list_drinks", "log_pomodoro"),
        registered=None,
        forbidden=("send_chat_message", "update_display"),
        post=None,
        clock=None,
    ):
        self.write_config(agents)
        active_definitions = [tool_definition(name) for name in active]
        return load_auxiliary_manager(
            enabled=True,
            config_path=self.config_path,
            tool_definitions=active_definitions,
            registered_tool_names=set(registered or active),
            forbidden_tool_names=set(forbidden),
            log_path=self.log_path,
            max_tool_calls_per_turn=10,
            post=post,
            monotonic=clock,
        )

    @staticmethod
    def agent(name, tools, **extra):
        return {
            "name": name,
            "base_url": f"http://{name}.test/v1",
            "model": f"{name}-model",
            "tools": tools,
            **extra,
        }

    def read_events(self):
        with open(self.log_path, encoding="utf-8") as handle:
            return [json.loads(line) for line in handle]

    def test_missing_disabled_and_empty_configs_are_noops_without_logs(self):
        missing_path = os.path.join(self.temp_dir.name, "missing.json")
        common = {
            "tool_definitions": [tool_definition("log_drink")],
            "registered_tool_names": {"log_drink"},
            "forbidden_tool_names": set(),
            "log_path": self.log_path,
            "max_tool_calls_per_turn": 10,
        }

        self.assertIsNone(load_auxiliary_manager(
            enabled=True,
            config_path=missing_path,
            **common,
        ))
        with open(self.config_path, "w", encoding="utf-8") as handle:
            handle.write("not json")
        self.assertIsNone(load_auxiliary_manager(
            enabled=False,
            config_path=self.config_path,
            **common,
        ))
        self.write_config([])
        self.assertIsNone(load_auxiliary_manager(
            enabled=True,
            config_path=self.config_path,
            **common,
        ))
        self.assertFalse(os.path.exists(self.log_path))

    def test_classifier_and_work_payloads_use_separate_parameters(self):
        post = QueuePost([
            classifier_response(True),
            work_response("log_drink", {"label": "water"}),
        ])
        manager = self.load([
            self.agent(
                "secretary",
                ["log_drink"],
                classification_timeout_seconds=3.5,
            )
        ], post=post)

        turn = manager.evaluate(
            [{"role": "user", "content": "Please log water."}],
            [tool_definition("log_drink")],
        )

        self.assertEqual(turn.agent_name, "secretary")
        self.assertEqual(turn.response["content"], "")
        self.assertEqual(turn.response["reasoning"], "")
        self.assertEqual(turn.response["tool_calls"][0]["name"], "log_drink")
        self.assertNotEqual(turn.response["tool_calls"][0]["id"], "provider_reused_id")

        classifier_call, work_call = post.calls
        classifier = classifier_call["json"]
        self.assertEqual(classifier_call["timeout"], 3.5)
        self.assertEqual(classifier["temperature"], 0.6)
        self.assertEqual(classifier["top_p"], 0.95)
        self.assertEqual(classifier["reasoning_effort"], "low")
        self.assertEqual(classifier["max_tokens"], 200)
        self.assertIs(classifier["cache_prompt"], True)

        work = work_call["json"]
        self.assertEqual(work["max_tokens"], 20_000)
        self.assertIs(work["cache_prompt"], True)
        self.assertNotIn("temperature", work)
        self.assertNotIn("top_p", work)
        self.assertNotIn("reasoning_effort", work)
        self.assertEqual(
            [item["function"]["name"] for item in work["tools"]],
            ["log_drink"],
        )

        events = self.read_events()
        self.assertEqual([event["event"] for event in events], ["classification", "work"])
        self.assertEqual(events[0]["request"]["payload"]["max_tokens"], 200)
        self.assertEqual(events[1]["request"]["payload"]["max_tokens"], 20_000)

    def test_first_agent_owns_overlapping_tools(self):
        manager = self.load([
            self.agent("first", ["log_drink", "list_drinks"]),
            self.agent("second", ["list_drinks", "log_pomodoro"]),
        ])

        self.assertEqual(manager.owner_for("log_drink"), "first")
        self.assertEqual(manager.owner_for("list_drinks"), "first")
        self.assertEqual(manager.owner_for("log_pomodoro"), "second")
        self.assertEqual(manager.agents[1].tools, ("log_pomodoro",))

    def test_known_inactive_mcp_tool_is_removed_without_starting_manager(self):
        manager = self.load(
            [self.agent("searcher", ["brave_web_search"])],
            active=("log_drink",),
            registered=("log_drink", "brave_web_search"),
        )

        self.assertIsNone(manager)
        self.assertFalse(os.path.exists(self.log_path))

    def test_malformed_schema_unknown_and_forbidden_tools_fail_fast(self):
        common = {
            "enabled": True,
            "config_path": self.config_path,
            "tool_definitions": [
                tool_definition("log_drink"),
                tool_definition("send_chat_message"),
            ],
            "registered_tool_names": {"log_drink", "send_chat_message"},
            "forbidden_tool_names": {"send_chat_message"},
            "log_path": self.log_path,
            "max_tool_calls_per_turn": 10,
        }

        with open(self.config_path, "w", encoding="utf-8") as handle:
            handle.write("{")
        with self.assertRaisesRegex(AuxiliaryConfigError, "malformed.*Valid tool names"):
            load_auxiliary_manager(**common)

        self.write_config([self.agent("bad", ["log_drink"])], schema_version=2)
        with self.assertRaisesRegex(AuxiliaryConfigError, "unsupported schema_version"):
            load_auxiliary_manager(**common)

        self.write_config([self.agent("bad", ["typo_tool"])])
        with self.assertRaises(AuxiliaryConfigError) as raised:
            load_auxiliary_manager(**common)
        self.assertIn("unknown tool", str(raised.exception))
        self.assertIn("log_drink", str(raised.exception))
        self.assertIn("send_chat_message", str(raised.exception))

        self.write_config([self.agent("bad", ["send_chat_message"])])
        with self.assertRaisesRegex(AuxiliaryConfigError, "non-delegatable"):
            load_auxiliary_manager(**common)

    def test_decline_escalates_and_strips_image_data(self):
        post = QueuePost([classifier_response(False)])
        manager = self.load(
            [self.agent("secretary", ["log_drink"])],
            post=post,
        )
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What do you think?"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/jpeg;base64,SECRET"},
                },
            ],
        }]

        turn = manager.evaluate(messages, [tool_definition("log_drink")])

        self.assertIsNone(turn.response)
        self.assertEqual(turn.escalation_packet["type"], "auxiliary_escalation")
        self.assertEqual(turn.escalation_packet["attempts"][0]["outcome"], "declined")
        self.assertNotIn("SECRET", json.dumps(post.calls[0]["json"]))
        self.assertNotIn("SECRET", json.dumps(turn.escalation_packet))
        self.assertEqual(self.read_events()[-1]["event"], "escalation")

    def test_failed_agent_backs_off_while_next_agent_keeps_serving(self):
        clock = FakeClock()
        calls = []

        def post(url, **kwargs):
            calls.append((url, kwargs))
            if "dead.test" in url:
                raise requests.ConnectionError("endpoint down")
            if "response_format" in kwargs["json"]:
                return FakeResponse(classifier_response(True))
            return FakeResponse(work_response("list_drinks"))

        manager = self.load([
            self.agent("dead", ["log_drink"]),
            self.agent("live", ["list_drinks"]),
        ], post=post, clock=clock)

        with patch.object(aux_agents, "info") as log_info:
            first = manager.evaluate(
                [{"role": "user", "content": "first event"}],
                [],
            )
            clock.advance(5)
            second = manager.evaluate(
                [{"role": "user", "content": "second event"}],
                [],
            )

        self.assertEqual(first.agent_name, "live")
        self.assertEqual(second.agent_name, "live")
        warnings = [
            call for call in log_info.call_args_list
            if "[AUX WARNING] dead unavailable" in call.args[0]
        ]
        self.assertEqual(len(warnings), 1)
        failures = [event for event in self.read_events() if event["event"] == "failure"]
        self.assertEqual(len(failures), 2)
        self.assertEqual(
            [event["operator_warning_emitted"] for event in failures],
            [True, False],
        )
        self.assertEqual([event["retry_in_seconds"] for event in failures], [5, 10])


class OrchestratorRoutingTests(unittest.TestCase):
    def make_orchestrator(self):
        orchestrator = Orchestrator.__new__(Orchestrator)
        orchestrator.running = True
        orchestrator.mcp_tools = []
        orchestrator.ctx = Context()
        orchestrator.ctx.add_system("system")
        orchestrator.ctx.add_user("Log the completed action.")
        orchestrator.ctx_lock = threading.Lock()
        orchestrator.ai = Mock()
        orchestrator.ui_state = Mock()
        orchestrator.vision_mode = "chill"
        orchestrator._llm_failures = 0
        orchestrator._build_turn_reminder = lambda: "CURRENT BOUNDARY"
        orchestrator._set_agent_state = lambda *args, **kwargs: None

        def execute_batch(_self, calls, *, record_results=False):
            _self.running = False
            if record_results:
                return [{
                    "id": calls[0]["id"],
                    "name": calls[0]["name"],
                    "arguments": calls[0]["arguments"],
                    "result": {"status": "ok"},
                }]
            return None

        orchestrator._execute_tool_batch = types.MethodType(
            execute_batch,
            orchestrator,
        )
        return orchestrator

    def test_auxiliary_tool_turn_skips_hosted_brain(self):
        orchestrator = self.make_orchestrator()
        response = {
            "content": "",
            "reasoning": "",
            "reasoning_details": [],
            "tool_calls": [{
                "id": "aux_unique",
                "name": "log_drink",
                "arguments": {"label": "water"},
            }],
        }
        orchestrator.aux_agents = Mock()
        orchestrator.aux_agents.evaluate.return_value = AuxiliaryTurn(
            response=response,
            agent_name="secretary",
            context_revision="revision",
            escalation_packet=None,
        )

        with (
            patch("main.get_tool_definitions", return_value=[tool_definition("log_drink")]),
            patch("main.play_sound"),
        ):
            Orchestrator._turn(orchestrator)

        orchestrator.ai.chat_with_tools.assert_not_called()
        orchestrator.aux_agents.record_tool_results.assert_called_once()

    def test_escalated_user_question_uses_hosted_brain(self):
        orchestrator = self.make_orchestrator()
        packet = {
            "schema_version": 1,
            "type": "auxiliary_escalation",
            "reason": "judgment required",
        }
        orchestrator.aux_agents = Mock()
        orchestrator.aux_agents.evaluate.return_value = AuxiliaryTurn(
            response=None,
            agent_name=None,
            context_revision="revision",
            escalation_packet=packet,
        )
        orchestrator.aux_agents.append_escalation_to_reminder.return_value = (
            "CURRENT BOUNDARY\nAUXILIARY ESCALATION PACKET"
        )
        orchestrator.ai.chat_with_tools.return_value = {
            "content": "",
            "reasoning": "",
            "reasoning_details": [],
            "tool_calls": [{
                "id": "host_call",
                "name": "send_chat_message",
                "arguments": {"text": "Hosted answer"},
            }],
        }

        tools = [tool_definition("send_chat_message")]
        with (
            patch("main.get_tool_definitions", return_value=tools),
            patch("main.play_sound"),
        ):
            Orchestrator._turn(orchestrator)

        orchestrator.ai.chat_with_tools.assert_called_once()
        sent_messages, sent_tools = orchestrator.ai.chat_with_tools.call_args.args
        self.assertEqual(sent_tools, tools)
        self.assertIn("AUXILIARY ESCALATION PACKET", sent_messages[-1]["content"])


if __name__ == "__main__":
    unittest.main()
