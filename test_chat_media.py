import base64
import json
import os
import sys
import tempfile
import types
import unittest

# Keep these stdlib-only tests runnable before optional project dependencies are
# installed. config.py only needs load_dotenv() for import-time environment setup.
try:
    import dotenv  # noqa: F401
except ModuleNotFoundError:
    dotenv_stub = types.ModuleType("dotenv")
    dotenv_stub.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = dotenv_stub

from chat_media import ChatMediaError, build_chat_message, media_data_from_message
import context as context_module
from context import Context


def data_url(mime: str, raw: bytes) -> str:
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


class ChatMediaTests(unittest.TestCase):
    def test_text_only_stays_a_string(self):
        content, attachments = build_chat_message("hello", [])
        self.assertEqual(content, "hello")
        self.assertEqual(attachments, [])

    def test_gif_is_preserved_as_multimodal_content(self):
        raw = b"GIF89a" + b"\x00" * 20
        content, attachments = build_chat_message(
            "",
            [{
                "name": "reaction.gif",
                "type": "image/gif",
                "data_url": data_url("image/gif", raw),
            }],
        )
        self.assertIn("animated GIF", content[0]["text"])
        self.assertEqual(content[1]["image_url"]["url"], data_url("image/gif", raw))
        self.assertEqual(attachments[0]["type"], "image/gif")

        msg = {"content": content, "_chat_images": attachments}
        self.assertEqual(
            media_data_from_message(msg, attachments[0]["id"]),
            ("image/gif", raw),
        )

    def test_spoofed_type_is_rejected(self):
        with self.assertRaises(ChatMediaError):
            build_chat_message(
                "look",
                [{
                    "name": "not-really.png",
                    "type": "image/png",
                    "data_url": data_url("image/png", b"GIF89a" + b"\x00" * 5),
                }],
            )

    def test_mime_mismatch_is_rejected(self):
        with self.assertRaises(ChatMediaError):
            build_chat_message(
                "look",
                [{
                    "name": "photo.jpg",
                    "type": "image/jpeg",
                    "data_url": data_url(
                        "image/png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 5
                    ),
                }],
            )

    def test_old_media_is_replaced_by_text_description(self):
        raw = b"GIF89a" + b"\x00" * 20
        content, attachments = build_chat_message(
            "This is the victory dance.",
            [{
                "name": "dance.gif",
                "type": "image/gif",
                "data_url": data_url("image/gif", raw),
            }],
        )
        ctx = Context()
        ctx.add_user(content, chat_images=attachments)
        self.assertTrue(
            ctx.note_latest_image_response("A person dances after finishing the task.")
        )
        for index in range(30):
            ctx.add_user(f"later message {index}")

        self.assertEqual(ctx.demote_old_images(keep_last=30), 1)
        demoted = ctx.messages[0]
        self.assertIsInstance(demoted["content"], str)
        self.assertIn("victory dance", demoted["content"])
        self.assertIn("dance.gif", demoted["content"])
        self.assertIn("A person dances", demoted["content"])
        self.assertFalse(ctx._is_image_message(demoted))

    def test_context_save_never_writes_raw_media(self):
        raw = b"GIF89a" + b"\x00" * 20
        content, attachments = build_chat_message(
            "Save the description, not the file.",
            [{
                "name": "private.gif",
                "type": "image/gif",
                "data_url": data_url("image/gif", raw),
            }],
        )
        ctx = Context()
        ctx.add_user(content, chat_images=attachments)
        ctx.note_latest_image_response("A small looping celebration animation.")

        with tempfile.TemporaryDirectory() as temp_dir:
            original_path = context_module.CONTEXT_FILE
            context_module.CONTEXT_FILE = os.path.join(temp_dir, "context.json")
            try:
                ctx.save()
                with open(context_module.CONTEXT_FILE, "r") as saved_file:
                    saved_text = saved_file.read()
                saved = json.loads(saved_text)
            finally:
                context_module.CONTEXT_FILE = original_path

        self.assertNotIn("base64", saved_text)
        self.assertNotIn(base64.b64encode(raw).decode(), saved_text)
        self.assertIn("small looping celebration", saved[0]["content"])


if __name__ == "__main__":
    unittest.main()
