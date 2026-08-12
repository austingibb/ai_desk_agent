import io
import json
import sys
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

try:
    import dotenv  # noqa: F401
except ModuleNotFoundError:
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = dotenv_stub

import ai_client
import camera
import config


VALID_PERCEPTION = {
    "people_present": {"status": "yes", "evidence": "one person at a desk"},
    "activity": {"category": "working", "description": "using a laptop"},
    "lighting": {
        "source": "mixed",
        "lamps": "on",
        "level": "normal",
        "evidence": "window and lamp light",
    },
    "notable_details": {"status": "none", "description": "nothing unusual"},
}


# Mirrors scene.PERCEPTION_SCHEMA's shape, including the module-level identity:
# build_perception_payload() hands this same object out by reference, which is
# what _add_requested_observations has to deep-copy before editing.
FAKE_PERCEPTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": list(VALID_PERCEPTION),
    "properties": {field: {"type": "object"} for field in VALID_PERCEPTION},
}


def _fake_scene_module():
    def build_perception_payload(*, model, image_data_uri, max_tokens):
        return {
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_uri}},
                    {"type": "text", "text": "canonical prompt"},
                ],
            }],
            "max_tokens": max_tokens,
            "enable_thinking": False,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "scene_perception",
                    "strict": True,
                    "schema": FAKE_PERCEPTION_SCHEMA,
                },
            },
        }

    def validate_perception(value):
        # scene.validate_perception checks for missing fields only, so extra
        # keys pass. The fake must match or it would reject the very field
        # _add_requested_observations exists to add.
        missing = set(VALID_PERCEPTION) - set(value)
        if missing:
            raise ValueError(f"perception is missing fields: {sorted(missing)}")

    return types.SimpleNamespace(
        build_perception_payload=build_perception_payload,
        validate_perception=validate_perception,
        __file__="fake/scene.py",
    )


class _Response:
    ok = True

    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        return None

    def json(self):
        return self._data


class VisionClientTests(unittest.TestCase):
    def test_aarg_payload_enables_gemma_thinking(self):
        fake_scene = _fake_scene_module()
        response = _Response({
            "choices": [{"message": {"content": json.dumps(VALID_PERCEPTION)}}],
            "usage": {"completion_tokens": 25},
            "timings": {"predicted_per_second": 20.0},
        })
        request_history = Mock()
        request_history.snapshot_for.return_value = ("canonical requests", "b" * 40, "aarg_mlx")
        description_log = Mock()

        with (
            patch.object(ai_client, "VISION_PROVIDER", "aarg_mlx"),
            patch.object(ai_client, "VISION_AARG_MLX_DIR", ""),
            patch.object(ai_client, "VISION_ENABLE_THINKING", True),
            patch.object(ai_client, "VISION_THINKING_BUDGET", 256),
            patch.object(ai_client, "VISION_MAX_TOKENS", 350),
            patch.object(ai_client.importlib, "import_module", return_value=fake_scene),
            patch.object(ai_client.requests, "post", return_value=response) as post,
        ):
            client = ai_client.VisionClient(request_history, description_log)
            result = client.describe(
                "data:image/jpeg;base64,AA==",
                source="main_camera_background",
                captured_at=1_785_800_000.0,
            )

        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["messages"][0]["content"][0]["type"], "image_url")
        self.assertIs(payload["enable_thinking"], True)
        self.assertEqual(payload["thinking_budget"], 256)
        self.assertEqual(payload["thinking_start_token"], "<|think|>")
        self.assertEqual(payload["thinking_end_token"], "<channel|>")
        # 256 thinking + 350 answer + 133 headroom for requested_observations
        self.assertEqual(payload["max_tokens"], 739)
        self.assertEqual(json.loads(result), VALID_PERCEPTION)
        self.assertEqual(client.last_request_commit, "b" * 40)
        description_log.append.assert_called_once()
        logged = description_log.append.call_args.kwargs
        self.assertEqual(logged["request_commit"], "b" * 40)
        self.assertEqual(logged["source"], "main_camera_background")

    def test_aarg_parser_accepts_exposed_thought_prefix(self):
        fake_scene = _fake_scene_module()
        with (
            patch.object(ai_client, "VISION_PROVIDER", "aarg_mlx"),
            patch.object(ai_client, "VISION_AARG_MLX_DIR", ""),
            patch.object(ai_client.importlib, "import_module", return_value=fake_scene),
        ):
            client = ai_client.VisionClient()

        parsed = client._parse_aarg_perception({
            "choices": [{
                "message": {
                    "content": "<|think|>inspect the frame<channel|>" + json.dumps(VALID_PERCEPTION)
                }
            }]
        })
        self.assertEqual(parsed, VALID_PERCEPTION)

    def _aarg_client(self, fake_scene):
        with (
            patch.object(ai_client, "VISION_PROVIDER", "aarg_mlx"),
            patch.object(ai_client, "VISION_AARG_MLX_DIR", ""),
            patch.object(ai_client.importlib, "import_module", return_value=fake_scene),
        ):
            return ai_client.VisionClient()

    def test_requests_content_gets_a_schema_field_to_land_in(self):
        client = self._aarg_client(_fake_scene_module())
        with (
            patch.object(ai_client, "VISION_ENABLE_THINKING", False),
            patch.object(ai_client, "VISION_MAX_TOKENS", 350),
        ):
            payload = client._build_aarg_payload(
                "data:image/jpeg;base64,AA==", "always report visible drinks"
            )

        schema = payload["response_format"]["json_schema"]["schema"]
        field = schema["properties"]["requested_observations"]
        self.assertIn("requested_observations", schema["required"])
        self.assertEqual(field["properties"]["status"]["enum"], ["present", "none", "unclear"])
        self.assertEqual(field["properties"]["description"]["maxLength"], 400)
        self.assertEqual(payload["max_tokens"], 483)

        text = payload["messages"][0]["content"][-1]["text"]
        self.assertIn("always report visible drinks", text)
        self.assertIn("requested_observations", text)

    def test_schema_edit_does_not_mutate_the_shared_canonical_schema(self):
        client = self._aarg_client(_fake_scene_module())
        client._build_aarg_payload("data:image/jpeg;base64,AA==", "look for mugs")
        self.assertNotIn("requested_observations", FAKE_PERCEPTION_SCHEMA["properties"])
        self.assertNotIn("requested_observations", FAKE_PERCEPTION_SCHEMA["required"])

    def test_empty_requests_file_leaves_the_canonical_payload_alone(self):
        client = self._aarg_client(_fake_scene_module())
        with (
            patch.object(ai_client, "VISION_ENABLE_THINKING", False),
            patch.object(ai_client, "VISION_MAX_TOKENS", 350),
        ):
            payload = client._build_aarg_payload("data:image/jpeg;base64,AA==", "   ")

        schema = payload["response_format"]["json_schema"]["schema"]
        self.assertNotIn("requested_observations", schema["properties"])
        self.assertEqual(payload["max_tokens"], 350)
        self.assertEqual(payload["messages"][0]["content"][-1]["text"], "canonical prompt")

    def test_parser_keeps_a_valid_requested_observations_field(self):
        client = self._aarg_client(_fake_scene_module())
        perception = dict(VALID_PERCEPTION)
        perception["requested_observations"] = {
            "status": "present",
            "description": "a full mug of coffee on the desk",
        }
        parsed = client._parse_aarg_perception({
            "choices": [{"message": {"content": json.dumps(perception)}}]
        })
        self.assertEqual(parsed["requested_observations"]["status"], "present")

    def test_parser_drops_a_malformed_requested_observations_field(self):
        client = self._aarg_client(_fake_scene_module())
        perception = dict(VALID_PERCEPTION)
        perception["requested_observations"] = {"status": "definitely", "description": 7}
        parsed = client._parse_aarg_perception({
            "choices": [{"message": {"content": json.dumps(perception)}}]
        })
        self.assertNotIn("requested_observations", parsed)
        self.assertEqual(parsed["people_present"], VALID_PERCEPTION["people_present"])

    def test_requests_written_for_another_mode_are_not_sent(self):
        fake_scene = _fake_scene_module()
        response = _Response({
            "choices": [{"message": {"content": json.dumps(VALID_PERCEPTION)}}],
        })
        request_history = Mock()
        # snapshot_for withholds the body and reports the mode it was written for.
        request_history.snapshot_for.return_value = ("", "c" * 40, "generic")

        with (
            patch.object(ai_client, "VISION_PROVIDER", "aarg_mlx"),
            patch.object(ai_client, "VISION_AARG_MLX_DIR", ""),
            patch.object(ai_client, "VISION_ENABLE_THINKING", False),
            patch.object(ai_client, "VISION_MAX_TOKENS", 350),
            patch.object(ai_client.importlib, "import_module", return_value=fake_scene),
            patch.object(ai_client.requests, "post", return_value=response) as post,
        ):
            client = ai_client.VisionClient(request_history)
            client.describe("data:image/jpeg;base64,AA==", max_retries=1)

        payload = post.call_args.kwargs["json"]
        schema = payload["response_format"]["json_schema"]["schema"]
        self.assertEqual(payload["messages"][0]["content"][-1]["text"], "canonical prompt")
        self.assertNotIn("requested_observations", schema["properties"])
        self.assertEqual(payload["max_tokens"], 350)
        self.assertEqual(client.requests_stale, "generic")

    def test_generic_provider_keeps_existing_text_first_payload(self):
        with patch.object(ai_client, "VISION_PROVIDER", "generic"):
            client = ai_client.VisionClient()
        payload = client._build_generic_payload("data:image/jpeg;base64,AA==")
        content = payload["messages"][0]["content"]
        self.assertEqual(content[0]["type"], "text")
        self.assertEqual(content[1]["type"], "image_url")


class CameraTests(unittest.TestCase):
    def test_encode_applies_mount_rotation_and_resize(self):
        source = Image.new("RGB", (20, 10), "red")
        with (
            patch.object(camera, "CAMERA_ROTATION", 90),
            patch.object(camera, "CAMERA_IMAGE_MAX_WIDTH", 5),
        ):
            jpeg, uri = camera._encode_image(source)
        with Image.open(io.BytesIO(jpeg)) as result:
            self.assertEqual(result.size, (5, 10))
        self.assertTrue(uri.startswith("data:image/jpeg;base64,"))

    def test_auto_backend_uses_opencv_on_macos(self):
        with (
            patch.object(camera, "CAMERA_BACKEND", "auto"),
            patch.object(camera.sys, "platform", "darwin"),
        ):
            self.assertEqual(camera._select_backend(), "opencv")

    def test_explicit_picamera_backend_is_preserved(self):
        with patch.object(camera, "CAMERA_BACKEND", "picamera2"):
            self.assertEqual(camera._select_backend(), "picamera2")

    def test_picamera_backend_retains_capture_contract(self):
        class Request:
            released = False

            def make_array(self, stream):
                if stream == "main":
                    return np.zeros((12, 20, 3), dtype=np.uint8)
                return np.zeros((180, 160), dtype=np.uint8)

            def release(self):
                self.released = True

        class Picamera2:
            stopped = False

            def create_still_configuration(self, **kwargs):
                return kwargs

            def configure(self, config):
                self.config = config

            def start(self):
                return None

            def capture_request(self):
                return Request()

            def stop(self):
                self.stopped = True

        module = types.SimpleNamespace(Picamera2=Picamera2)
        with (
            patch.dict(sys.modules, {"picamera2": module}),
            patch.object(camera, "CAMERA_ROTATION", 0),
        ):
            pi_camera = camera._PiCamera()
            jpeg, uri = pi_camera.capture()
            lores = pi_camera.capture_lores()
            pi_camera.close()

        self.assertTrue(jpeg.startswith(b"\xff\xd8"))
        self.assertTrue(uri.startswith("data:image/jpeg;base64,"))
        self.assertEqual(lores.shape, (120, 160))
        self.assertEqual(lores.dtype, np.float32)
        self.assertTrue(pi_camera.picam.stopped)


class ConfigPromptTests(unittest.TestCase):
    def test_web_search_disabled_removes_mcp_prompt(self):
        with patch.object(config, "ENABLE_WEB_SEARCH", False):
            prompt = config.build_system_prompt()
        self.assertNotIn("Brave Search", prompt)
        self.assertNotIn("brave_web_search", prompt)
        self.assertIn("cannot verify time-sensitive outside information", prompt)
        self.assertIn("Web search is disabled in this runtime", prompt)

    def test_web_search_enabled_includes_mcp_prompt(self):
        with patch.object(config, "ENABLE_WEB_SEARCH", True):
            prompt = config.build_system_prompt()
        self.assertIn("Brave Search", prompt)
        self.assertIn("brave_web_search", prompt)

    def test_web_search_disabled_skips_mcp_initialization(self):
        import main

        with (
            patch.object(main, "ENABLE_CAMERA", False),
            patch.object(main, "ENABLE_REOLINK", False),
            patch.object(main, "ENABLE_WEB_SEARCH", False),
            patch.object(main, "MCPClient") as mcp_client,
            patch.object(main.signal, "signal"),
        ):
            orchestrator = main.Orchestrator()

        mcp_client.assert_not_called()
        self.assertIsNone(orchestrator.mcp)
        self.assertEqual(orchestrator.mcp_tools, [])


if __name__ == "__main__":
    unittest.main()
