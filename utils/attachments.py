import asyncio
import base64
import math
import mimetypes
import os
from io import BytesIO
from pathlib import Path
from PIL import Image, ImageOps
import discord
import logging
from config import (
    ALLOWED_IMAGE_EXTENSIONS, MAX_IMAGE_ATTACHMENTS, MAX_IMAGE_MB, TARGET_IMAGE_PIXELS,
    JPEG_QUALITY, ATTACHMENT_TEXT_EXTENSIONS, MAX_TEXT_ATTACHMENTS, MAX_TOTAL_TEXT_ATTACHMENT_CHARS,
    MAX_TEXT_ATTACHMENT_MB, MAX_TEXT_ATTACHMENT_CHARS
)
from services.extractors import extract_text_from_file

log = logging.getLogger("rag-bot")

# Read the vision flag once at startup
VISION_ENABLED = os.getenv("ENABLE_VISION", "true").lower() == "true"

def get_image_mime(filename: str) -> str:
    mime, _ = mimetypes.guess_type(filename)
    if mime and mime.startswith("image/"): return mime
    ext = Path(filename).suffix.lower()
    return {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".gif": "image/gif", ".webp": "image/webp"}.get(ext, "image/png")

def image_has_meaningful_alpha(image: Image.Image) -> bool:
    if image.mode in ("RGBA", "LA"): return True
    if image.mode == "P" and "transparency" in image.info: return True
    return False

def resize_image_to_target(image: Image.Image) -> Image.Image:
    width, height = image.size
    current_pixels = width * height
    if current_pixels <= TARGET_IMAGE_PIXELS: return image
    scale = math.sqrt(TARGET_IMAGE_PIXELS / float(current_pixels))
    new_width, new_height = max(1, int(width * scale)), max(1, int(height * scale))
    while new_width * new_height > TARGET_IMAGE_PIXELS:
        if new_width >= new_height and new_width > 1: new_width -= 1
        elif new_height > 1: new_height -= 1
        else: break
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

def encode_standardized_image(image: Image.Image) -> tuple[bytes, str]:
    max_bytes = MAX_IMAGE_MB * 1024 * 1024
    if image_has_meaningful_alpha(image):
        img = image.convert("RGBA")
        buffer = BytesIO()
        img.save(buffer, format="PNG", optimize=True)
        data = buffer.getvalue()
        if len(data) <= max_bytes: return data, "image/png"
    img = image.convert("RGB")
    quality = JPEG_QUALITY
    while True:
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=quality, optimize=True)
        data = buffer.getvalue()
        if len(data) <= max_bytes or quality <= 35: return data, "image/jpeg"
        quality -= 10

def process_image_bytes(raw: bytes) -> tuple[bytes, str]:
    image = Image.open(BytesIO(raw))
    image.load()
    image = ImageOps.exif_transpose(image)
    image = resize_image_to_target(image)
    return encode_standardized_image(image)

async def collect_image_attachments(message: discord.Message) -> tuple[list[dict], list[str]]:
    image_blocks, warnings = [], []
    if not message.attachments:
        return image_blocks, warnings

    # --- VISION GUARDRAIL ---
    if not VISION_ENABLED:
        has_image = any(
            (att.content_type and att.content_type.startswith("image/")) or
            Path(att.filename or "").suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
            for att in message.attachments
        )
        if has_image:
            warnings.append(
                "Image attachments were ignored because the current model does not support vision (ENABLE_VISION=false).")
        return image_blocks, warnings

    for attachment in message.attachments:
        filename = attachment.filename or "attachment.png"
        ext = Path(filename).suffix.lower()
        content_type = attachment.content_type or ""
        if not (ext in ALLOWED_IMAGE_EXTENSIONS or content_type.startswith("image/")):
            continue
        if len(image_blocks) >= MAX_IMAGE_ATTACHMENTS:
            warnings.append("Additional image attachments were ignored.")
            break
        raw = await attachment.read()
        try:
            data, mime = await asyncio.to_thread(process_image_bytes, raw)
        except Exception:
            log.exception("Failed processing image attachment: %s", filename)
            warnings.append(f"`{filename}` could not be processed as an image.")
            continue
        if len(data) > MAX_IMAGE_MB * 1024 * 1024:
            warnings.append(f"`{filename}` was still too large after resizing and was ignored.")
            continue
        encoded = base64.b64encode(data).decode("utf-8")
        image_blocks.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}})
    return image_blocks, warnings

async def collect_text_attachments(message: discord.Message) -> tuple[list[str], list[str]]:
    text_blocks, warnings = [], []
    remaining_chars = MAX_TOTAL_TEXT_ATTACHMENT_CHARS
    if not message.attachments: return text_blocks, warnings
    for attachment in message.attachments:
        filename = attachment.filename or "attachment.txt"
        ext = Path(filename).suffix.lower()
        content_type = attachment.content_type or ""
        if not (ext in ATTACHMENT_TEXT_EXTENSIONS or content_type.startswith("text/")): continue
        if len(text_blocks) >= MAX_TEXT_ATTACHMENTS or remaining_chars <= 0:
            warnings.append("Additional text/document attachments were ignored.")
            break
        if MAX_TEXT_ATTACHMENT_MB > 0 and attachment.size > MAX_TEXT_ATTACHMENT_MB * 1024 * 1024:
            warnings.append(f"`{filename}` ignored: larger than {MAX_TEXT_ATTACHMENT_MB} MB.")
            continue
        raw = await attachment.read()
        try:
            text = await asyncio.to_thread(extract_text_from_file, filename, raw)
        except Exception:
            text = raw.decode("utf-8", errors="ignore")
        text = text.strip()
        if not text: warnings.append(f"`{filename}` contained no readable text."); continue
        allowed_chars = min(MAX_TEXT_ATTACHMENT_CHARS, remaining_chars)
        if len(text) > allowed_chars: text = text[:allowed_chars] + "\n\n[Attachment truncated because it was too long.]"
        remaining_chars -= len(text)
        text_blocks.append(f"Attached file name: {filename}\nAttached file contents:\n{text}")
    return text_blocks, warnings