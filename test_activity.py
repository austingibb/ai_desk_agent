import json
import os
import tempfile
import unittest

from activity import ActivityStore, classify_scene
from main import ChatUIState, Orchestrator


def observation(activity: str, confidence: str = "high") -> dict:
    presence = "AK" if activity == "working_at_computer" else "AFK"
    return {
        "presence": presence,
        "activity": activity,
        "confidence": confidence,
        "evidence": f"evidence for {activity}",
    }


class ActivityClassificationTests(unittest.TestCase):
    def test_structured_computer_work_is_ak(self):
        value = classify_scene(json.dumps({
            "people_present": {"status": "yes", "evidence": "one person"},
            "activity": {
                "category": "working",
                "description": "typing on a laptop at the desk",
            },
        }))
        self.assertEqual(value["presence"], "AK")
        self.assertEqual(value["activity"], "working_at_computer")

    def test_eating_at_computer_is_afk_eating(self):
        value = classify_scene(
            "A person is eating a sandwich while typing on the laptop."
        )
        self.assertEqual(value["presence"], "AFK")
        self.assertEqual(value["activity"], "eating")

    def test_empty_room_is_afk_out(self):
        value = classify_scene(json.dumps({
            "people_present": {"status": "no", "evidence": "room is empty"},
            "activity": {"category": "none", "description": ""},
        }))
        self.assertEqual(value["presence"], "AFK")
        self.assertEqual(value["activity"], "out")

    def test_specific_afk_categories_are_detected(self):
        examples = {
            "Someone is asleep in bed.": "sleeping",
            "A person is doing yoga on the floor.": "exercising",
            "The man is folding laundry near the bed.": "chores",
            "A woman is relaxing on the couch watching TV.": "relaxing",
        }
        for description, expected in examples.items():
            with self.subTest(description=description):
                value = classify_scene(description)
                self.assertEqual(value["presence"], "AFK")
                self.assertEqual(value["activity"], expected)

    def test_vague_presence_is_low_confidence_and_no_evidence_is_ignored(self):
        vague = classify_scene("A person is visible near the doorway.")
        self.assertEqual(vague["activity"], "relaxing")
        self.assertEqual(vague["confidence"], "low")
        self.assertIsNone(classify_scene("The lights are on and the desk is tidy."))


class ActivityStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.temp_dir.name, "activity.json")

    def tearDown(self):
        self.temp_dir.cleanup()

    def store(self, **kwargs):
        return ActivityStore(
            file_path=self.path,
            retention_seconds=kwargs.pop("retention_seconds", 9000),
            confirmation_observations=kwargs.pop("confirmation_observations", 2),
            **kwargs,
        )

    def test_segments_extend_and_transition_with_monotonic_timestamps(self):
        store = self.store()
        first = store.observe(
            observation("working_at_computer"), observed_at=100, source="background"
        )
        self.assertTrue(first["changed"])

        same = store.observe(
            observation("working_at_computer"), observed_at=120, source="on_demand"
        )
        self.assertFalse(same["changed"])
        self.assertEqual(same["current"]["last_observed_at"], 120)

        changed = store.observe(
            observation("eating"), observed_at=130, source="background"
        )
        self.assertTrue(changed["changed"])
        self.assertEqual(changed["current"]["activity"], "eating")
        values = store.list_recent()
        self.assertEqual(values[0]["ended_at"], 130)
        self.assertIsNone(values[1]["ended_at"])

    def test_low_confidence_transition_requires_confirmation(self):
        store = self.store(confirmation_observations=2)
        store.observe(observation("working_at_computer"), observed_at=100)

        first = store.observe(observation("relaxing", "low"), observed_at=110)
        self.assertFalse(first["changed"])
        self.assertEqual(first["current"]["activity"], "working_at_computer")

        second = store.observe(observation("relaxing", "low"), observed_at=120)
        self.assertTrue(second["changed"])
        self.assertEqual(second["current"]["activity"], "relaxing")

    def test_state_and_pending_confirmation_survive_restart(self):
        store = self.store(confirmation_observations=2)
        store.observe(observation("working_at_computer"), observed_at=100)
        store.observe(observation("relaxing", "low"), observed_at=110)

        restored = self.store(confirmation_observations=2)
        self.assertEqual(restored.current()["activity"], "working_at_computer")
        result = restored.observe(
            observation("relaxing", "low"), observed_at=120
        )
        self.assertTrue(result["changed"])
        self.assertEqual(result["current"]["activity"], "relaxing")

    def test_invalid_hierarchy_is_ignored(self):
        store = self.store()
        result = store.observe({
            "presence": "AK",
            "activity": "eating",
            "confidence": "high",
            "evidence": "invalid combination",
        }, observed_at=100)
        self.assertTrue(result["ignored"])
        self.assertIsNone(store.current())

    def test_closed_segments_outside_retention_are_pruned(self):
        store = self.store(retention_seconds=50)
        store.observe(observation("working_at_computer"), observed_at=100)
        store.observe(observation("out"), observed_at=110)
        store.observe(observation("eating"), observed_at=200)
        values = store.list_recent()
        self.assertEqual(
            [item["activity"] for item in values], ["out", "eating"]
        )

    def test_current_segment_remains_visible_inside_a_short_lookback(self):
        store = self.store()
        store.observe(observation("sleeping"), observed_at=100)

        values = store.list_recent(since_hours=1, now=10_000)

        self.assertEqual([item["activity"] for item in values], ["sleeping"])


class ActivityIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.store = ActivityStore(
            file_path=os.path.join(self.temp_dir.name, "activity.json")
        )
        self.orchestrator = Orchestrator.__new__(Orchestrator)
        self.orchestrator.activity_store = self.store
        self.orchestrator.ui_state = ChatUIState(activity_enabled=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_camera_description_updates_store_and_live_ui(self):
        self.orchestrator._record_activity(
            "A person is eating breakfast while seated at the computer.",
            captured_at=100,
            source="background",
        )

        current = self.store.current()
        self.assertEqual(current["presence"], "AFK")
        self.assertEqual(current["activity"], "eating")
        activity = self.orchestrator.ui_state.snapshot()["activity"]
        self.assertEqual(activity["presence"], "AFK")
        self.assertEqual(activity["activity"], "eating")

    def test_read_tool_returns_persisted_segments(self):
        self.store.observe(observation("working_at_computer"), observed_at=100)
        self.store.observe(observation("out"), observed_at=130)

        result = self.orchestrator._tool_list_activity({"limit": 10})

        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["segments"]), 2)
        self.assertEqual(result["current"]["activity"], "out")


if __name__ == "__main__":
    unittest.main()
