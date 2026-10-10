import asyncio
import logging
import re

import discord
from pydantic_ai import ImageUrl

from config import CONVERSATION_BASE_WINDOW, CONVERSATION_MAX_WINDOW, CONVERSATION_MAX_HISTORY_IMAGES
from core.db import get_talk_window_start, set_talk_window_start
from utils.attachments import collect_image_attachments, _is_image_attachment
from utils.formatting import clean_answer_text

log = logging.getLogger("rag-bot")

_HISTORY_RETRIES = 3
_RETRY_DELAY_SECONDS = 1.5
_PART_PREFIX_RE = re.compile(r"^Part \d+/\d+\s*\n?")


def build_server_channel_list(guild: discord.Guild) -> str:
    if guild is None: return "(Not in a server)"
    channels = []
    for ch in guild.text_channels:
        perms = ch.permissions_for(guild.me)
        if perms.view_channel and perms.read_messages:
            topic = f" - {ch.topic}" if ch.topic else ""
            channels.append(f"#{ch.name}{topic}")
    if not channels: return "(No accessible channels)"
    return "\n".join(f"- {c}" for c in channels)


def _author_label(msg: discord.Message) -> str:
    author = msg.author
    if isinstance(author, discord.Member) and author.nick:
        display_name = author.nick
    elif getattr(author, "global_name", None):
        display_name = author.global_name
    else:
        display_name = author.name
    return f"{display_name} (@{author.name})"


def _message_body(msg: discord.Message, bot_user_id: int, text_override: str | None = None,
                  ignore_attachment_ids: set[int] | None = None) -> str:
    ignore = ignore_attachment_ids or set()
    text = text_override if text_override is not None else msg.clean_content
    if msg.author.id == bot_user_id and text_override is None:
        text = _PART_PREFIX_RE.sub("", clean_answer_text(text))
    parts = [text] if text else []
    attachments = [att for att in msg.attachments if att.id not in ignore]
    has_image = any(_is_image_attachment(att) for att in attachments)
    has_file = any(not _is_image_attachment(att) for att in attachments)
    if has_image:
        parts.append("[image attached]")
    if has_file:
        parts.append("[file attached]")
    if not parts:
        parts.append("(no content)")
    return "\n".join(parts)


def format_turn_line(msg: discord.Message, bot_user_id: int, text_override: str | None = None,
                     ignore_attachment_ids: set[int] | None = None) -> str:
    """Unified metadata line, used for BOTH historical messages and the current
    trigger message so the next turn's prompt prefix matches byte-for-byte."""
    ts = msg.created_at.strftime("%Y-%m-%d %H:%M UTC")
    header = f"[{ts} | id:{msg.id}] {_author_label(msg)}"
    ref = msg.reference
    if ref and ref.message_id:
        header += f" (in response to id:{ref.message_id})"
    return f"{header}: {_message_body(msg, bot_user_id, text_override, ignore_attachment_ids)}"


def _is_eligible(msg: discord.Message, bot_user_id: int) -> bool:
    """Window eligibility: skip other bots and messages with no content and
    no attachments."""
    if msg.author.bot and msg.author.id != bot_user_id:
        return False
    if not msg.clean_content and not msg.attachments:
        return False
    return True


async def _resolve_referenced_message(ref: discord.MessageReference | None,
                                      channel: discord.abc.Messageable) -> discord.Message | None:
    """Resolve a message reference to a live Message, or None when the
    reference is empty, points to a deleted message, or cannot be fetched."""
    if not ref or not ref.message_id:
        return None
    target = ref.resolved
    if isinstance(target, discord.Message):
        return target
    if target is not None:
        return None
    try:
        return await channel.fetch_message(ref.message_id)
    except Exception:
        return None


async def resolve_replied_message(message: discord.Message) -> discord.Message | None:
    """Resolve the message this message directly replies to, if any."""
    return await _resolve_referenced_message(message.reference, message.channel)


async def _fetch_recent_pool(message: discord.Message) -> list[discord.Message]:
    """Latest CONVERSATION_MAX_WINDOW messages before the trigger, oldest first.
    Retries on transient Discord 5xx errors and degrades to an empty pool."""
    for attempt in range(_HISTORY_RETRIES):
        try:
            pool: list[discord.Message] = [
                msg async for msg in message.channel.history(limit=CONVERSATION_MAX_WINDOW, before=message)
            ]
            pool.reverse()
            return pool
        except discord.DiscordServerError as e:
            if attempt < _HISTORY_RETRIES - 1:
                delay = _RETRY_DELAY_SECONDS * (attempt + 1)
                log.warning("History fetch failed (attempt %d/%d): %s. Retrying in %.1fs.",
                            attempt + 1, _HISTORY_RETRIES, e, delay)
                await asyncio.sleep(delay)
            else:
                log.warning("History fetch failed after %d attempts; continuing without conversation history: %s",
                            _HISTORY_RETRIES, e)
    return []


async def _bot_md_substitute(msg: discord.Message) -> tuple[str | None, set[int]]:
    """When one of our own replies was too long for Discord and went out as an
    .md attachment, recover the full answer text so history stays complete."""
    for att in msg.attachments:
        if not (att.filename or "").lower().endswith(".md"):
            continue
        try:
            raw = await att.read()
        except Exception as e:
            log.warning("Could not download bot md attachment %s: %s", att.filename, e)
            continue
        text = raw.decode("utf-8", errors="ignore").strip()
        if text:
            return text, {att.id}
    return None, set()

async def _message_turn(msg: discord.Message, bot_user_id: int, include_images: bool) -> tuple[str, str | list]:
    role = "assistant" if msg.author.id == bot_user_id else "user"
    if role == "assistant":
        override, ignore_ids = await _bot_md_substitute(msg)
    else:
        override, ignore_ids = None, set()
    line = format_turn_line(msg, bot_user_id, override, ignore_ids)
    if role == "assistant" or not include_images:
        return role, line
    blocks, _ = await collect_image_attachments(msg)
    if not blocks:
        return role, line
    return role, [line, *[ImageUrl(url=b["image_url"]["url"]) for b in blocks]]


async def build_conversation_turns(message: discord.Message, bot_user_id: int) -> list[tuple[str, str | list]]:
    """Stepping-window conversation history as (role, content) turns.

    The window grows append-only from CONVERSATION_BASE_WINDOW up to
    CONVERSATION_MAX_WINDOW messages. Only when the persisted watermark
    falls outside the freshly fetched pool is the window reset to the
    newest CONVERSATION_BASE_WINDOW messages (one deliberate cache
    invalidation), and the watermark advanced to the new window start.
    """
    pool = await _fetch_recent_pool(message)
    if not pool:
        return []
    channel_id = message.channel.id
    watermark = await get_talk_window_start(channel_id)
    if watermark and any(msg.id == watermark for msg in pool):
        window = [msg for msg in pool if msg.id >= watermark]
    else:
        window = pool[-CONVERSATION_BASE_WINDOW:]
        channel_name = getattr(message.channel, "name", "") or "Direct Message"
        await set_talk_window_start(channel_id, window[0].id, channel_name)
        log.info("Talk window reset for channel %s: watermark moved to message %s (%d message window).",
                 channel_id, window[0].id, len(window))
    window = [msg for msg in window if _is_eligible(msg, bot_user_id)]
    if not window:
        return []
    image_budget = CONVERSATION_MAX_HISTORY_IMAGES
    include_images: dict[int, bool] = {}
    for msg in reversed(window):
        if image_budget <= 0:
            break
        if any(_is_image_attachment(att) for att in msg.attachments):
            include_images[msg.id] = True
            image_budget -= 1
    turns: list[tuple[str, str | list]] = []
    for msg in window:
        role, content = await _message_turn(msg, bot_user_id, include_images.get(msg.id, False))
        turns.append((role, content))
    return turns