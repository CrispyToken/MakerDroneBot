import discord
from utils.formatting import clean_answer_text
from config import CONVERSATION_MAX_CHAIN_EXTRA


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
    if target is not None:
        return f" (in reply to {target.author.display_name})"
    return " (in reply to an earlier message not shown here)"


def _format_line(msg: discord.Message, bot_user_id: int, known: dict[int, discord.Message]) -> str:
    return (f"[{_format_ts(msg)}] {msg.author.display_name}"
            f"{_reply_suffix(msg, known)}: {_message_text(msg, bot_user_id)}")


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

    target = ref.resolved
    if target is None:
        try:
            target = await message.channel.fetch_message(ref.message_id)
        except Exception:
            return []
    if target is None or target.author.id != bot_user_id:
        return []  # only replies to our own out-of-window messages are traced

    extras: list[discord.Message] = []
    current = target
    while True:
        if current is None or current.id in known or any(m.id == current.id for m in extras):
            break
        if len(extras) >= max_extra:
            break
        extras.append(current)

        # current is a bot answer; find the user message that triggered it.
        cref = current.reference
        if not cref or not cref.message_id:
            break
        user_msg = cref.resolved
        if user_msg is None:
            try:
                user_msg = await message.channel.fetch_message(cref.message_id)
            except Exception:
                break
        if user_msg is None or user_msg.id in known or any(m.id == user_msg.id for m in extras):
            break  # chain reconnected with visible context
        if len(extras) >= max_extra:
            break
        extras.append(user_msg)

        # Continue only if that user message was itself a reply to our bot.
        uref = user_msg.reference
        if not uref or not uref.message_id:
            break  # initial user prompt reached; it stays as the last extra
        next_bot = uref.resolved
        if next_bot is None:
            try:
                next_bot = await message.channel.fetch_message(uref.message_id)
            except Exception:
                break
        if next_bot is None or next_bot.author.id != bot_user_id:
            break
        current = next_bot

    extras.reverse()  # oldest -> newest
    return extras


async def get_conversation_context(message: discord.Message, bot_user_id: int, bot_display_name: str,
                                   max_history: int = 15) -> list[str]:
    window: list[discord.Message] = []
    if max_history > 0:
        async for msg in message.channel.history(limit=max_history * 3, before=message):
            if not _is_eligible(msg, bot_user_id):
                continue
            window.append(msg)
            if len(window) >= max_history:
                break
    window.reverse()          # oldest -> newest, ends just before the trigger

    known: dict[int, discord.Message] = {m.id: m for m in window}

    extras = await _trace_outside_reply_chain(message, bot_user_id, known, CONVERSATION_MAX_CHAIN_EXTRA)
    if extras:
        for m in extras:
            known[m.id] = m
        window = extras + window

    return [_format_line(m, bot_user_id, known) for m in window]