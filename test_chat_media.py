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

from chat_media import (
    ChatMediaError,
    build_chat_message,
    media_data_from_message,
    merge_queued_messages,
    queued_message_payload,
    update_queued_text,
)
import context as context_module
from context import Context


def data_url(mime: str, raw: bytes) -> str:
    return f"data:{mime};base64,{base64.b64encode(raw).decode()}"


def png_bytes(suffix: bytes = b"") -> bytes:
    return b"\x89PNG\r\n\x1a\n" + suffix


class ChatMediaTests(unittest.TestCase):
    def _queued(self, queue_id, text, images):
        content, attachments = build_chat_message(text, images)
        return {
            "id": queue_id,
            "created_at": 1.0,
            "text": text,
            "content": content,
            "chat_images": attachments,
        }

    def test_text_only_stays_a_string(self):
        content, attachments = build_chat_message("hello", [])
        self.assertEqual(content, "hello")
        self.assertEqual(attachments, [])

    def test_gif_is_rejected(self):
        raw = b"GIF89a" + b"\x00" * 20
        with self.assertRaisesRegex(ChatMediaError, "static PNG, JPEG, or WebP"):
            build_chat_message(
                "",
                [{
                    "name": "reaction.gif",
                    "type": "image/gif",
                    "data_url": data_url("image/gif", raw),
                }],
            )

    def test_png_is_preserved_as_multimodal_content(self):
        raw = png_bytes(b"\x00" * 20)
        content, attachments = build_chat_message(
            "",
            [{
                "name": "reaction.png",
                "type": "image/png",
                "data_url": data_url("image/png", raw),
            }],
        )
        self.assertIn("1 image", content[0]["text"])
        self.assertEqual(content[1]["image_url"]["url"], data_url("image/png", raw))
        self.assertEqual(attachments[0]["type"], "image/png")

        msg = {"content": content, "_chat_images": attachments}
        self.assertEqual(
            media_data_from_message(msg, attachments[0]["id"]),
            ("image/png", raw),
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
        raw = png_bytes(b"\x00" * 20)
        content, attachments = build_chat_message(
            "This is the victory dance.",
            [{
                "name": "dance.png",
                "type": "image/png",
                "data_url": data_url("image/png", raw),
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
        self.assertIn("dance.png", demoted["content"])
        self.assertIn("A person dances", demoted["content"])
        self.assertFalse(ctx._is_image_message(demoted))

    def test_context_save_never_writes_raw_media(self):
        raw = png_bytes(b"\x00" * 20)
        content, attachments = build_chat_message(
            "Save the description, not the file.",
            [{
                "name": "private.png",
                "type": "image/png",
                "data_url": data_url("image/png", raw),
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

    def test_queue_merge_keeps_newest_images_and_reindexes(self):
        raws = [png_bytes(bytes([index]) * 12) for index in range(6)]
        entries = [
            self._queued(
                "q_1",
                "first",
                [{
                    "name": f"{index}.png",
                    "type": "image/png",
                    "data_url": data_url("image/png", raws[index]),
                } for index in range(3)],
            ),
            self._queued(
                "q_2",
                "correction",
                [{
                    "name": f"{index}.png",
                    "type": "image/png",
                    "data_url": data_url("image/png", raws[index]),
                } for index in range(3, 6)],
            ),
        ]

        content, attachments, visible, dropped = merge_queued_messages(
            entries, max_images=4, max_media_bytes=10_000
        )

        self.assertEqual(dropped, 2)
        self.assertEqual([a["name"] for a in attachments], ["2.png", "3.png", "4.png", "5.png"])
        self.assertEqual([a["content_index"] for a in attachments], [1, 2, 3, 4])
        self.assertEqual(content[0]["type"], "text")
        self.assertIn("first\ncorrection", content[0]["text"])
        self.assertIn("2 images were dropped", visible)

    def test_queue_merge_byte_limit_and_singular_note(self):
        raw_old = png_bytes(b"a" * 30)
        raw_new = png_bytes(b"b" * 12)
        entries = [
            self._queued("q_1", "old", [{
                "name": "old.png", "type": "image/png",
                "data_url": data_url("image/png", raw_old),
            }]),
            self._queued("q_2", "new", [{
                "name": "new.png", "type": "image/png",
                "data_url": data_url("image/png", raw_new),
            }]),
        ]

        _, attachments, visible, dropped = merge_queued_messages(
            entries, max_images=4, max_media_bytes=len(raw_new)
        )

        self.assertEqual(dropped, 1)
        self.assertEqual([a["name"] for a in attachments], ["new.png"])
        self.assertIn("1 image was dropped", visible)

    def test_undo_payload_contains_original_text_and_data_urls(self):
        raw = png_bytes(b"x" * 12)
        entry = self._queued("q_1", "typo", [{
            "name": "undo.png", "type": "image/png",
            "data_url": data_url("image/png", raw),
        }])
        payload = queued_message_payload(entry)
        self.assertEqual(payload["text"], "typo")
        self.assertEqual(payload["images"][0]["data_url"], data_url("image/png", raw))

    def test_empty_text_edit_requires_attachments(self):
        entry = self._queued("q_1", "hello", [])
        with self.assertRaisesRegex(ChatMediaError, "Use Undo send instead"):
            update_queued_text(entry, "")

        raw = png_bytes(b"x" * 12)
        media_entry = self._queued("q_2", "hello", [{
            "name": "keep.png", "type": "image/png",
            "data_url": data_url("image/png", raw),
        }])
        update_queued_text(media_entry, "")
        self.assertEqual(media_entry["text"], "")
        self.assertIn("The user shared", media_entry["content"][0]["text"])


if __name__ == "__main__":
    unittest.main()
