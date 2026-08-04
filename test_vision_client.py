import io
import json
import sys
import types
import unittest
from unittest.mock import patch

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
            "response_format": {"type": "json_schema"},
        }

    def validate_perception(value):
        if set(value) != set(VALID_PERCEPTION):
            raise ValueError("invalid perception")

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

        with (
            patch.object(ai_client, "VISION_PROVIDER", "aarg_mlx"),
            patch.object(ai_client, "VISION_AARG_MLX_DIR", ""),
            patch.object(ai_client, "VISION_ENABLE_THINKING", True),
            patch.object(ai_client, "VISION_THINKING_BUDGET", 256),
            patch.object(ai_client, "VISION_MAX_TOKENS", 350),
            patch.object(ai_client.importlib, "import_module", return_value=fake_scene),
            patch.object(ai_client.requests, "post", return_value=response) as post,
        ):
            client = ai_client.VisionClient()
            result = client.describe("data:image/jpeg;base64,AA==")

        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["messages"][0]["content"][0]["type"], "image_url")
        self.assertIs(payload["enable_thinking"], True)
        self.assertEqual(payload["thinking_budget"], 256)
        self.assertEqual(payload["thinking_start_token"], "<|think|>")
        self.assertEqual(payload["thinking_end_token"], "<channel|>")
        self.assertEqual(payload["max_tokens"], 606)
        self.assertEqual(json.loads(result), VALID_PERCEPTION)

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
