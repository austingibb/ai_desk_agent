import json
import os
import tempfile
import unittest
from unittest.mock import patch

import context as context_module
from context import Context


class ContextIdentityTests(unittest.TestCase):
    def test_new_messages_receive_unique_stable_ids(self):
        ctx = Context()
        first = ctx.add_user("same")
        second = ctx.add_user("same")

        self.assertNotEqual(first, second)
        self.assertEqual(first, ctx.messages[0]["_chat_id"])
        self.assertEqual(second, ctx.messages[1]["_chat_id"])
        self.assertTrue(first.startswith("c_"))

    def test_legacy_load_backfills_and_persists_ids_once(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = os.path.join(temp_dir, "context.json")
            with open(path, "w") as saved:
                json.dump([{"role": "user", "content": "legacy", "_ts": 1.0}], saved)

            original = context_module.CONTEXT_FILE
            context_module.CONTEXT_FILE = path
            try:
                first = Context()
                original_save = Context.save
                with patch.object(
                    Context,
                    "save",
                    autospec=True,
                    side_effect=lambda instance: original_save(instance),
                ) as save:
                    self.assertTrue(first.load())
                    backfilled = first.messages[0]["_chat_id"]

                    second = Context()
                    self.assertTrue(second.load())
                    self.assertEqual(second.messages[0]["_chat_id"], backfilled)
                    self.assertEqual(save.call_count, 1)
            finally:
                context_module.CONTEXT_FILE = original

    def test_compaction_summary_receives_an_id(self):
        ctx = Context()
        ctx.messages = [ctx.new_message("system", "prompt")]
        ctx.messages.extend(ctx.new_message("user", f"message {index}") for index in range(4))

        ctx._apply_compact(
            ctx.messages[0], [], 3, 2, 2, "range", "summary", 1.0, 2.0
        )

        summary = next(message for message in ctx.messages if ctx._is_summary(message))
        self.assertTrue(summary["_chat_id"].startswith("c_"))


if __name__ == "__main__":
    unittest.main()
