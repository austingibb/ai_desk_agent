import json
import os
import subprocess
import tempfile
import unittest

from vision_history import VisionDescriptionLog, VisionRequestHistory


class VisionRequestHistoryTests(unittest.TestCase):
    def test_migrates_and_versions_only_requests_file(self):
        with tempfile.TemporaryDirectory() as directory:
            repo_dir = os.path.join(directory, "requests_for_image_model")
            legacy = os.path.join(directory, "requests_for_image_model.md")
            with open(legacy, "w", encoding="utf-8") as handle:
                handle.write("# Requests for Image Model\n\nLook for mugs.\n")

            history = VisionRequestHistory(repo_dir=repo_dir, legacy_file=legacy)
            first_commit = history.current_commit()

            self.assertFalse(os.path.exists(legacy))
            self.assertIn("Look for mugs.", history.read())
            self.assertEqual(len(first_commit), 40)
            tracked = subprocess.run(
                ["git", "ls-files"],
                cwd=repo_dir,
                check=True,
                text=True,
                capture_output=True,
            ).stdout.splitlines()
            remotes = subprocess.run(
                ["git", "remote"],
                cwd=repo_dir,
                check=True,
                text=True,
                capture_output=True,
            ).stdout.strip()
            self.assertEqual(tracked, ["requests_for_image_model.md"])
            self.assertEqual(remotes, "")

            second_commit = history.update("Look for mugs and water bottles.")
            self.assertNotEqual(first_commit, second_commit)
            self.assertIn("water bottles", history.read())
            self.assertEqual(history.read().count("# Requests for Image Model"), 1)

            headed_commit = history.update(
                "# Requests for Image Model\n\nLook for mugs and glasses."
            )
            self.assertNotEqual(second_commit, headed_commit)
            self.assertEqual(history.read().count("# Requests for Image Model"), 1)

            with open(history.file_path, "a", encoding="utf-8") as handle:
                handle.write("Manual note.\n")
            snapshot, third_commit = history.snapshot()
            self.assertNotEqual(headed_commit, third_commit)
            self.assertIn("Manual note.", snapshot)

    def test_unchanged_update_keeps_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            history = VisionRequestHistory(
                repo_dir=os.path.join(directory, "history"),
                legacy_file=None,
            )
            first = history.update("Watch the desk.")
            second = history.update("Watch the desk.")
            self.assertEqual(first, second)


class VisionDescriptionLogTests(unittest.TestCase):
    def test_appends_description_with_request_commit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "vision_logs", "descriptions.jsonl")
            log = VisionDescriptionLog(path)
            record = log.append(
                description="One person is working at a desk.",
                request_commit="a" * 40,
                model="gemma-test",
                provider="aarg_mlx",
                source="main_camera_background",
                captured_at=1_785_800_000.0,
                latency_s=8.125,
                usage={"completion_tokens": 20},
                timings={"predicted_per_second": 30.0},
            )

            with open(path, encoding="utf-8") as handle:
                saved = json.loads(handle.readline())
            self.assertEqual(saved, record)
            self.assertEqual(saved["request_commit"], "a" * 40)
            self.assertEqual(saved["description"], "One person is working at a desk.")
            self.assertEqual(saved["source"], "main_camera_background")


if __name__ == "__main__":
    unittest.main()
