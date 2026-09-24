import json
import threading
import unittest
from unittest.mock import Mock, patch

from camera_manager import CameraFeed, CameraManager


class CameraManagerTests(unittest.TestCase):
    def manager(self, modes=("cache", "cache"), **kwargs):
        feeds = []
        for index, mode in enumerate(modes):
            device = Mock(spec=["capture", "close"])
            device.capture.return_value = (f"jpeg-{index}".encode(), f"uri-{index}")
            feeds.append(CameraFeed(f"cam{index}", device, mode, .02))
        vision = Mock(last_request_commit="commit")
        vision.describe.side_effect = lambda uri, **kw: f"scene:{uri}"
        return CameraManager(feeds, vision, **kwargs)

    def test_separate_caches_and_source_metadata(self):
        manager = self.manager()
        for camera_id in manager.feeds:
            manager.capture(camera_id)
        self.assertEqual(manager.read("cam0")["description"], "scene:uri-0")
        self.assertEqual(manager.read("cam1")["description"], "scene:uri-1")
        self.assertEqual(manager.feeds["cam1"].jpeg, b"jpeg-1")
        self.assertEqual(manager.read("cam1")["source"], "cam1_on_demand")
        self.assertEqual(manager.read()["camera_id"], "cam0")
        self.assertEqual(manager.vision.describe.call_count, 2)

    def test_poll_only_reads_are_fresh_and_never_cached(self):
        manager = self.manager(("poll_only",))
        manager.start()
        self.assertEqual(manager.threads, [])
        for _ in range(2):
            self.assertEqual(manager.read()["status"], "ok")
        self.assertEqual(manager.vision.describe.call_count, 2)
        self.assertIsNone(manager.feeds["cam0"].scene)
        self.assertIsNone(manager.feeds["cam0"].jpeg)
        manager.close()
        manager.feeds["cam0"].device.close.assert_called_once()

    def test_cache_reads_are_instant_and_force_capture_refreshes(self):
        manager = self.manager(("cache",))
        self.assertEqual(manager.read()["status"], "error")
        manager.vision.describe.assert_not_called()
        self.assertEqual(manager.read(fresh=True)["status"], "ok")
        manager.read()
        manager.vision.describe.assert_called_once()
        manager.read(fresh=True)
        self.assertEqual(manager.vision.describe.call_count, 2)

    def test_failed_refresh_retains_last_good_cache_and_age(self):
        manager = self.manager(("cache",))
        with patch("camera_manager.time.time", return_value=100):
            manager.capture("cam0")
        manager.vision.describe.side_effect = RuntimeError("offline")
        self.assertIsNone(manager.capture("cam0"))
        with patch("camera_manager.time.time", return_value=130):
            result = manager.read()
        self.assertEqual(result["age_seconds"], 30)
        self.assertEqual(result["description"], "scene:uri-0")
        self.assertEqual(manager.feeds["cam0"].jpeg, b"jpeg-0")

    def test_background_refreshes_each_camera_without_motion(self):
        updated = threading.Event()
        counts = {"cam0": 0, "cam1": 0}
        def on_scene(scene, background):
            self.assertTrue(background)
            counts[scene["camera_id"]] += 1
            if min(counts.values()) >= 2:
                updated.set()
        manager = self.manager(on_scene=on_scene)
        manager.start()
        try:
            self.assertTrue(updated.wait(2), counts)
        finally:
            manager.close()
        self.assertTrue(all(not thread.is_alive() for thread in manager.threads))
        for feed in manager.feeds.values():
            feed.device.close.assert_called_once()

    def test_failing_camera_does_not_stop_other_updates(self):
        updated = threading.Event()
        manager = self.manager(on_scene=lambda *args: updated.set())
        manager.feeds["cam0"].device.capture.side_effect = RuntimeError("offline")
        manager.start()
        try:
            self.assertTrue(updated.wait(2))
            self.assertEqual(manager.read("cam1")["status"], "ok")
        finally:
            manager.close()

    def test_inference_metadata_stays_with_capture_across_threads(self):
        manager = self.manager()
        entered = threading.Event()
        release = threading.Event()
        def describe(uri, **kwargs):
            if uri == "uri-0":
                entered.set()
                self.assertTrue(release.wait(2))
            manager.vision.last_request_commit = uri
            return uri
        manager.vision.describe.side_effect = describe
        first = threading.Thread(target=manager.capture, args=("cam0",))
        second = threading.Thread(target=manager.capture, args=("cam1",))
        first.start()
        self.assertTrue(entered.wait(2))
        second.start()
        # The second camera must capture only after the first inference finishes.
        manager.feeds["cam1"].device.capture.assert_not_called()
        release.set()
        first.join(2)
        second.join(2)
        self.assertEqual(manager.read("cam0")["vision_request_commit"], "uri-0")
        self.assertEqual(manager.read("cam1")["vision_request_commit"], "uri-1")

    def test_motion_state_expires_without_affecting_other_cameras(self):
        manager = self.manager(chill_timeout=300)
        manager.feeds["cam1"].last_motion = 100
        with patch("camera_manager.time.monotonic", return_value=200):
            self.assertTrue(manager.motion_active)
        with patch("camera_manager.time.monotonic", return_value=401):
            self.assertFalse(manager.motion_active)

    def test_invalid_ids_and_modes(self):
        manager = self.manager()
        self.assertEqual(manager.read("missing")["available_cameras"], ["cam0", "cam1"])
        with self.assertRaises(ValueError):
            CameraFeed("cam", Mock(), "invalid")
        with self.assertRaises(ValueError):
            CameraManager([manager.feeds["cam0"]] * 2, Mock())


class CameraIntegrationTests(unittest.TestCase):
    def test_reolink_only_has_generic_tools_and_prompt(self):
        import config
        with patch.object(config, "CAMERAS_JSON", ""), patch.object(config, "ENABLE_CAMERA", False), patch.object(config, "ENABLE_REOLINK", True):
            tools = {t["function"]["name"] for t in config.get_tool_definitions()}
            self.assertTrue({"take_photo", "capture_photo", "update_vision_requests"} <= tools)
            self.assertIn("reolink (cache", config.build_system_prompt())

    def test_registry_validation_and_modes(self):
        import config
        cameras = [{"id": "desk", "type": "local", "mode": "poll_only"},
                   {"id": "door", "type": "reolink", "interval": 60},
                   {"id": "hall", "type": "reolink", "interval": 90}]
        with patch.object(config, "CAMERAS_JSON", json.dumps(cameras)):
            self.assertEqual(len(config.camera_settings(False, False)), 3)
        for invalid in [cameras + [cameras[0]], [{"id": "../bad", "type": "local"}],
                        [{"id": "desk", "type": "local", "interval": 0}],
                        [{"id": "desk", "type": "local", "mode": "bad"}]]:
            with patch.object(config, "CAMERAS_JSON", json.dumps(invalid)):
                with self.assertRaises(ValueError):
                    config.camera_settings()
        with patch.object(config, "CAMERAS_JSON", "[]"):
            self.assertNotIn("take_photo", {t["function"]["name"] for t in config.get_tool_definitions()})

    def test_background_updates_preserve_both_viewpoints(self):
        from main import Orchestrator
        orch = Orchestrator.__new__(Orchestrator)
        orch.cameras = Mock(feeds={"desk": None, "door": None})
        orch.scene_lock = threading.Lock()
        orch.pending_scenes = {}
        orch.motion_event = threading.Event()
        orch._record_activity = Mock()
        for camera_id in orch.cameras.feeds:
            orch._camera_scene_updated({"camera_id": camera_id, "description": camera_id,
                                        "timestamp": 100, "source": camera_id}, True)
        self.assertTrue(orch.motion_event.is_set())
        update = orch._consume_camera_updates()
        self.assertIn("[desk at", update)
        self.assertIn("[door at", update)
        self.assertFalse(orch.motion_event.is_set())
        orch._record_activity.assert_called_once()


if __name__ == "__main__":
    unittest.main()
