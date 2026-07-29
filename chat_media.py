"""Validation and message construction for user-posted chat images and GIFs."""

import base64
import binascii
import hashlib
import re

from config import CHAT_MAX_IMAGES_PER_MESSAGE, CHAT_MAX_MEDIA_BYTES


ALLOWED_IMAGE_TYPES = {
    "image/gif",
    "image/jpeg",
    "image/png",
    "image/webp",
}

_DATA_URI_RE = re.compile(
    r"^data:(image/(?:gif|jpeg|jpg|png|webp));base64,([A-Za-z0-9+/]*={0,2})$",
    re.IGNORECASE,
)


class ChatMediaError(ValueError):
    """A user-facing upload validation error."""

    def __init__(self, message: str, status_code: int = 400):
        self.status_code = status_code
        super().__init__(message)


def _normalized_mime(value: str) -> str:
    mime = (value or "").lower()
    return "image/jpeg" if mime == "image/jpg" else mime


def _has_valid_signature(mime: str, data: bytes) -> bool:
    if mime == "image/gif":
        return data.startswith((b"GIF87a", b"GIF89a"))
    if mime == "image/jpeg":
        return data.startswith(b"\xff\xd8\xff")
    if mime == "image/png":
        return data.startswith(b"\x89PNG\r\n\x1a\n")
    if mime == "image/webp":
        return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    return False


def build_chat_message(message: str, images) -> tuple[str | list, list]:
    """Validate uploads and return (OpenAI content, UI attachment metadata).

    Text-only messages remain strings for backward compatibility. Media messages
    use OpenAI/OpenRouter multipart content with text first, followed by one
    ``image_url`` block per uploaded image.
    """
    message = (message or "").strip()
    images = images or []
    if not isinstance(images, list):
        raise ChatMediaError("images must be a list")
    if len(images) > CHAT_MAX_IMAGES_PER_MESSAGE:
        raise ChatMediaError(
            f"Too many images (maximum {CHAT_MAX_IMAGES_PER_MESSAGE}).",
            status_code=413,
        )
    if not images:
        if not message:
            raise ChatMediaError("Empty message")
        return message, []

    content = []
    attachments = []
    total_bytes = 0

    for index, item in enumerate(images):
        if not isinstance(item, dict):
            raise ChatMediaError(f"Image {index + 1} is invalid.")

        supplied_mime = _normalized_mime(str(item.get("type", "")))
        data_url = item.get("data_url", "")
        if not isinstance(data_url, str):
            raise ChatMediaError(f"Image {index + 1} has no valid data URL.")

        match = _DATA_URI_RE.fullmatch(data_url)
        if not match:
            raise ChatMediaError(
                f"Image {index + 1} must be a PNG, JPEG, WebP, or GIF."
            )
        data_mime = _normalized_mime(match.group(1))
        if data_mime not in ALLOWED_IMAGE_TYPES:
            raise ChatMediaError(f"Unsupported image type: {data_mime}")
        if supplied_mime and supplied_mime != data_mime:
            raise ChatMediaError(f"Image {index + 1} type does not match its data.")

        encoded = match.group(2)
        # Reject obviously oversized input before allocating the decoded copy.
        estimated_bytes = (len(encoded) * 3) // 4
        if total_bytes + estimated_bytes > CHAT_MAX_MEDIA_BYTES:
            raise ChatMediaError(
                f"Attachments exceed the {CHAT_MAX_MEDIA_BYTES // (1024 * 1024)} MB limit.",
                status_code=413,
            )
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError):
            raise ChatMediaError(f"Image {index + 1} contains invalid base64 data.")
        if not raw or not _has_valid_signature(data_mime, raw):
            raise ChatMediaError(f"Image {index + 1} has an invalid file signature.")

        total_bytes += len(raw)
        if total_bytes > CHAT_MAX_MEDIA_BYTES:
            raise ChatMediaError(
                f"Attachments exceed the {CHAT_MAX_MEDIA_BYTES // (1024 * 1024)} MB limit.",
                status_code=413,
            )

        # Rebuild the URI from validated bytes so only canonical media types and
        # base64 reach OpenRouter.
        normalized_url = (
            f"data:{data_mime};base64,{base64.b64encode(raw).decode('ascii')}"
        )
        name = str(item.get("name", "")).strip()[:200]
        media_id = hashlib.sha256(raw).hexdigest()
        attachments.append(
            {
                "id": media_id,
                "name": name or f"image-{index + 1}",
                "type": data_mime,
                "content_index": index + 1,
                "size": len(raw),
            }
        )
        content.append(
            {"type": "image_url", "image_url": {"url": normalized_url}}
        )

    if not message:
        gifs = sum(1 for a in attachments if a["type"] == "image/gif")
        stills = len(attachments) - gifs
        labels = []
        if stills:
            labels.append(f"{stills} image{'s' if stills != 1 else ''}")
        if gifs:
            labels.append(f"{gifs} animated GIF{'s' if gifs != 1 else ''}")
        message = f"The user shared {' and '.join(labels)}. Respond to what they shared."

    content.insert(0, {"type": "text", "text": message})
    return content, attachments


def media_data_from_message(msg: dict, media_id: str) -> tuple[str, bytes] | None:
    """Resolve an authenticated chat-media ID from a live context message."""
    attachments = msg.get("_chat_images", [])
    content = msg.get("content", [])
    if not isinstance(attachments, list) or not isinstance(content, list):
        return None
    for attachment in attachments:
        if attachment.get("id") != media_id:
            continue
        index = attachment.get("content_index")
        if not isinstance(index, int) or not (0 <= index < len(content)):
            return None
        part = content[index]
        try:
            data_url = part["image_url"]["url"]
        except (KeyError, TypeError):
            return None
        match = _DATA_URI_RE.fullmatch(data_url)
        if not match:
            return None
        mime = _normalized_mime(match.group(1))
        try:
            raw = base64.b64decode(match.group(2), validate=True)
        except (binascii.Error, ValueError):
            return None
        if hashlib.sha256(raw).hexdigest() != media_id:
            return None
        return mime, raw
    return None
