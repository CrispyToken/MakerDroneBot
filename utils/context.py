import asyncio
import logging
import discord
from utils.formatting import clean_answer_text
from config import CONVERSATION_MAX_CHAIN_EXTRA

log = logging.getLogger("rag-bot")

_HISTORY_RETRIES = 3
_RETRY_DELAY_SECONDS = 1.5


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

def _format_ts(msg: discord.Message) -> str:
    return msg.created_at.strftime("%Y-%m-%d %H:%M UTC")


def _is_eligible(msg: discord.Message, bot_user_id: int) -> bool:
    """Window eligibility: skip other bots and messages with no content and
    no attachments."""
    if msg.author.bot and msg.author.id != bot_user_id:
        return False
    if not msg.clean_content and not msg.attachments:
        return False
    return True


def _message_text(msg: discord.Message, bot_user_id: int) -> str:
    content = msg.clean_content
    if not content and msg.attachments:
        has_image = any(att.content_type and att.content_type.startswith("image/") for att in msg.attachments)
        content = "[User attached an image]" if has_image else "[User attached a file]"
    if not content:
        content = "(no content)"
    if msg.author.id == bot_user_id:
        content = clean_answer_text(content)
    return content


def _reply_suffix(msg: discord.Message, known: dict[int, discord.Message]) -> str:
    ref = msg.reference
    if not ref or not ref.message_id:
        return ""
    target = ref.resolved or known.get(ref.message_id)
    if isinstance(target, discord.Message):
        return f" (in reply to {target.author.display_name})"
    if target is not None:
        return " (in reply to a deleted message)"
    return " (in reply to an earlier message not shown here)"


def _format_line(msg: discord.Message, bot_user_id: int, known: dict[int, discord.Message]) -> str:
    return (f"[{_format_ts(msg)}] {msg.author.display_name}"
            f"{_reply_suffix(msg, known)}: {_message_text(msg, bot_user_id)}")

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

async def _trace_outside_reply_chain(message: discord.Message, bot_user_id: int,
                                     known: dict[int, discord.Message],
                                     max_extra: int) -> list[discord.Message]:
    """Edge case: the trigger replies to one of OUR bot's messages that is
    not in the context window. Trace that reply thread backwards — bot answer,
    the user message that triggered it, and so on — until the first user
    prompt that is not itself a reply to the bot.

    Returns the extra messages oldest -> newest, ready to be prepended.
    """
    if max_extra <= 0:
        return []
    ref = message.reference
    if not ref or not ref.message_id or ref.message_id in known:
        return []

    target = await _resolve_referenced_message(ref, message.channel)
    if target is None or target.author.id != bot_user_id:
        return []

    extras: list[discord.Message] = []
    current = target
    while True:
        if current is None or current.id in known or any(m.id == current.id for m in extras):
            break
        if len(extras) >= max_extra:
            break
        extras.append(current)

        # current is a bot answer; find the user message that triggered it.
        user_msg = await _resolve_referenced_message(current.reference, message.channel)
        if user_msg is None or user_msg.id in known or any(m.id == user_msg.id for m in extras):
            break
        if len(extras) >= max_extra:
            break
        extras.append(user_msg)

        # Continue only if that user message was itself a reply to our bot.
        next_bot = await _resolve_referenced_message(user_msg.reference, message.channel)
        if next_bot is None or next_bot.author.id != bot_user_id:
            break
        current = next_bot

    extras.reverse()  # oldest -> newest
    return extras


async def _fetch_history_window(message: discord.Message, bot_user_id: int,
                                max_history: int) -> list[discord.Message]:
    """Fetch the recent-message window, retrying on transient Discord 5xx errors.

    Returns an empty window if Discord keeps failing after all retries, so a
    temporary API outage degrades gracefully instead of crashing the turn.
    """
    for attempt in range(_HISTORY_RETRIES):
        try:
            window: list[discord.Message] = []
            async for msg in message.channel.history(limit=max_history * 3, before=message):
                if not _is_eligible(msg, bot_user_id):
                    continue
                window.append(msg)
                if len(window) >= max_history:
                    break
            window.reverse()
            return window
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


async def get_conversation_context(message: discord.Message, bot_user_id: int, bot_display_name: str,
                                   max_history: int = 15) -> list[str]:
    window: list[discord.Message] = []
    if max_history > 0:
        window = await _fetch_history_window(message, bot_user_id, max_history)
    known: dict[int, discord.Message] = {m.id: m for m in window}

    extras = await _trace_outside_reply_chain(message, bot_user_id, known, CONVERSATION_MAX_CHAIN_EXTRA)
    if extras:
        for m in extras:
            known[m.id] = m
        window = extras + window

    return [_format_line(m, bot_user_id, known) for m in window]