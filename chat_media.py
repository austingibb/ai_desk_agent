"""Validation and message construction for user-posted static chat images."""

import base64
import binascii
import copy
import hashlib
import re

from config import CHAT_MAX_IMAGES_PER_MESSAGE, CHAT_MAX_MEDIA_BYTES


ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}

_DATA_URI_RE = re.compile(
    r"^data:(image/(?:jpeg|jpg|png|webp));base64,([A-Za-z0-9+/]*={0,2})$",
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
    if mime == "image/jpeg":
        return data.startswith(b"\xff\xd8\xff")
    if mime == "image/png":
        return data.startswith(b"\x89PNG\r\n\x1a\n")
    if mime == "image/webp":
        return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP"
    return False


def _attachment_fallback_text(attachments: list) -> str:
    count = len(attachments)
    return f"The user shared {count} image{'s' if count != 1 else ''}. Respond to what they shared."


def chat_content_text(content) -> str:
    """Return the model-facing text from string or multipart chat content."""
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    return "\n".join(
        str(part.get("text", "")).strip()
        for part in content
        if isinstance(part, dict)
        and part.get("type") == "text"
        and str(part.get("text", "")).strip()
    )


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
                f"Image {index + 1} must be a static PNG, JPEG, or WebP image."
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
        message = _attachment_fallback_text(attachments)

    content.insert(0, {"type": "text", "text": message})
    return content, attachments


def update_queued_text(entry: dict, message: str) -> None:
    """Update only the editable text of a validated queue record in place."""
    message = (message or "").strip()
    attachments = entry.get("chat_images", [])
    if not message and not attachments:
        raise ChatMediaError(
            "A message without attachments cannot be empty. Use Undo send instead."
        )

    entry["text"] = message
    model_text = message or _attachment_fallback_text(attachments)
    content = entry.get("content", "")
    if isinstance(content, list):
        updated = copy.deepcopy(content)
        for part in updated:
            if isinstance(part, dict) and part.get("type") == "text":
                part["text"] = model_text
                break
        else:
            updated.insert(0, {"type": "text", "text": model_text})
        entry["content"] = updated
    else:
        entry["content"] = model_text


def _attachment_data_url(entry: dict, attachment: dict) -> str | None:
    content = entry.get("content", [])
    index = attachment.get("content_index")
    if not isinstance(content, list) or not isinstance(index, int):
        return None
    if not (0 <= index < len(content)):
        return None
    try:
        value = content[index]["image_url"]["url"]
    except (KeyError, TypeError):
        return None
    return value if isinstance(value, str) and _DATA_URI_RE.fullmatch(value) else None


def queued_message_payload(entry: dict) -> dict:
    """Return the one-round-trip composer payload for Undo."""
    images = []
    for attachment in entry.get("chat_images", []):
        if not isinstance(attachment, dict):
            continue
        data_url = _attachment_data_url(entry, attachment)
        if not data_url:
            continue
        images.append({
            "name": attachment.get("name", "image"),
            "type": attachment.get("type", ""),
            "data_url": data_url,
        })
    return {"text": str(entry.get("text", "")), "images": images}


def _attachment_size(entry: dict, attachment: dict) -> int:
    size = attachment.get("size")
    if isinstance(size, int) and size >= 0:
        return size
    data_url = _attachment_data_url(entry, attachment)
    match = _DATA_URI_RE.fullmatch(data_url or "")
    if not match:
        return 0
    try:
        return len(base64.b64decode(match.group(2), validate=True))
    except (binascii.Error, ValueError):
        return 0


def merge_queued_messages(
    entries: list[dict],
    max_images: int = CHAT_MAX_IMAGES_PER_MESSAGE,
    max_media_bytes: int = CHAT_MAX_MEDIA_BYTES,
) -> tuple[str | list, list, str, int]:
    """Merge queued records, keeping the newest attachments within the limits.

    Returns ``(model_content, chat_images, visible_text, dropped_count)``.
    """
    model_texts = []
    visible_texts = []
    media = []

    for entry in entries:
        model_text = chat_content_text(entry.get("content", ""))
        if model_text:
            model_texts.append(model_text)
        visible_text = str(entry.get("text", "")).strip()
        if visible_text:
            visible_texts.append(visible_text)
        for attachment in entry.get("chat_images", []):
            if not isinstance(attachment, dict):
                continue
            data_url = _attachment_data_url(entry, attachment)
            if not data_url:
                continue
            media.append({
                "part": {"type": "image_url", "image_url": {"url": data_url}},
                "attachment": copy.deepcopy(attachment),
                "size": _attachment_size(entry, attachment),
            })

    dropped = 0
    total_bytes = sum(item["size"] for item in media)
    while media and (len(media) > max_images or total_bytes > max_media_bytes):
        removed = media.pop(0)
        total_bytes -= removed["size"]
        dropped += 1

    model_text = "\n".join(model_texts)
    visible_text = "\n".join(visible_texts)
    if dropped:
        noun = "image was" if dropped == 1 else "images were"
        note = (
            f"[{dropped} {noun} dropped because queued messages were merged "
            "and exceeded the per-message attachment limits.]"
        )
        model_text = "\n".join(part for part in (model_text, note) if part)
        visible_text = "\n".join(part for part in (visible_text, note) if part)

    attachments = []
    parts = [{"type": "text", "text": model_text}]
    for index, item in enumerate(media, start=1):
        attachment = item["attachment"]
        attachment["content_index"] = index
        attachment["size"] = item["size"]
        attachments.append(attachment)
        parts.append(item["part"])

    content = parts if attachments else model_text
    return content, attachments, visible_text, dropped


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
