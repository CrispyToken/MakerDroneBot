import re
import logging
import discord
from config import CHANNEL_READ_WINDOW_SIZE, CHANNEL_READ_MAX_IMAGES
from utils.attachments import collect_image_attachments

log = logging.getLogger("rag-bot")

DISCORD_MSG_LINK_RE = re.compile(r"^https?://(?:discord.com|discordapp.com)/channels/(\d+|@me)/(\d+)/(\d+)$")
_MAX_OUTPUT_CHARS = 10000
_MAX_SINGLE_MSG_CHARS = 1000


def _author_label(msg: discord.Message) -> str:
    author = msg.author
    if isinstance(author, discord.Member) and author.nick:
        display_name = author.nick
    elif getattr(author, "global_name", None):
        display_name = author.global_name
    else:
        display_name = author.name
    return f"{display_name} (@{author.name})"


def _serialize_embed(embed: discord.Embed) -> str:
    parts = []
    if embed.title:
        parts.append(f"[Embed Title] {embed.title}")
    if embed.description:
        parts.append(f"[Embed Description] {embed.description}")
    if embed.fields:
        for field in embed.fields:
            parts.append(f"[Embed Field] {field.name}: {field.value}")
    if embed.footer and embed.footer.text:
        parts.append(f"[Embed Footer] {embed.footer.text}")
    if embed.author and embed.author.name:
        parts.append(f"[Embed Author] {embed.author.name}")
    return "\n".join(parts)


def _message_line(msg: discord.Message, target_id: int | None) -> str:
    prefix = ">>> TARGET MESSAGE (linked by user) <<<\n" if (target_id is not None and msg.id == target_id) else ""
    ts = msg.created_at.strftime("%Y-%m-%d %H:%M UTC")

    content_parts = []
    if msg.clean_content:
        content_parts.append(msg.clean_content)
    for embed in msg.embeds:
        embed_text = _serialize_embed(embed)
        if embed_text:
            content_parts.append(embed_text)

    content = "\n".join(content_parts) if content_parts else "(no text content)"
    if len(content) > _MAX_SINGLE_MSG_CHARS:
        content = content[:_MAX_SINGLE_MSG_CHARS] + "... [truncated]"

    return f"{prefix}[{ts}] {_author_label(msg)}: {content}"


async def _append_message_parts(parts: list, msg: discord.Message, target_id: int | None,
                               include_images: bool, budget: list[int]) -> None:
    parts.append(_message_line(msg, target_id))
    if not include_images or not msg.attachments:
        return
    blocks, warnings = await collect_image_attachments(msg)
    ts = msg.created_at.strftime("%Y-%m-%d %H:%M UTC")
    for warning in warnings:
        parts.append(f"[{ts}] {warning}")
    if not blocks:
        return
    take = blocks[:budget[0]]
    budget[0] -= len(take)
    parts.extend(take)
    dropped = len(blocks) - len(take)
    if dropped:
        parts.append(f"[{ts}] {dropped} image(s) from {_author_label(msg)} omitted (channel-read image cap reached).")


async def _build_window_parts(window: list[discord.Message], target_id: int | None,
                              images_for: str) -> list:
    """Assemble transcript parts oldest -> newest, then cap by total length.

    Parts are grouped per message so a message's text and its images are kept
    or dropped together. When the combined transcript exceeds the output
    budget, the OLDEST messages are dropped first. This preserves the newest
    content, which is what a channel read is asking about, and for a
    message-link read it also keeps the linked (target) message, since only
    the older context ahead of it is removed.
    """
    groups: list[list] = []
    budget = [CHANNEL_READ_MAX_IMAGES]
    target_group = -1
    for index, msg in enumerate(window):
        include = images_for == "window" or (images_for == "target" and msg.id == target_id)
        msg_parts: list = []
        await _append_message_parts(msg_parts, msg, target_id, include, budget)
        if target_id is not None and msg.id == target_id:
            target_group = index
        groups.append(msg_parts)

    def _text_len(parts: list) -> int:
        return sum(len(p) for p in parts if isinstance(p, str))

    total = sum(_text_len(g) for g in groups)
    keep = [True] * len(groups)
    index = 0
    while total > _MAX_OUTPUT_CHARS and index < len(groups):
        if index != target_group:
            total -= _text_len(groups[index])
            keep[index] = False
        index += 1
    index = len(groups) - 1
    while total > _MAX_OUTPUT_CHARS and index >= 0:
        if index != target_group:
            total -= _text_len(groups[index])
            keep[index] = False
        index -= 1

    parts: list = []
    dropped = 0
    for kept, msg_parts in zip(keep, groups):
        if kept:
            parts.extend(msg_parts)
        else:
            dropped += 1
    if dropped:
        parts.insert(0, f"[Transcript truncated: {dropped} older message(s) omitted due to length.]")
    return parts


async def read_channel_content(bot: discord.Client, guild_id: int, channel_input: str,
                               message_link: str) -> list:
    """Returns an ordered list of str transcript lines and OpenAI-style image
    blocks ({"type": "image_url", ...}); the caller converts blocks for the model."""
    if message_link:
        match = DISCORD_MSG_LINK_RE.match(message_link.strip())
        if not match:
            return ["The provided message link is invalid or not recognized."]
        g_id, c_id, m_id = match.groups()
        if g_id == "@me":
            return ["The provided link is a direct message link, which is invalid for this tool."]
        if int(g_id) != guild_id:
            return ["The provided link points to a different server, which is invalid for this tool."]
        channel = bot.get_channel(int(c_id))
        if channel is None:
            try:
                channel = await bot.fetch_channel(int(c_id))
            except discord.NotFound:
                return ["The channel in the provided link could not be found."]
            except discord.Forbidden:
                return ["The bot does not have access to the channel in the provided link."]
            except Exception:
                return ["An error occurred while accessing the channel in the provided link."]
        if not isinstance(channel, discord.abc.Messageable):
            return ["The provided link points to a channel type that cannot be read."]
        try:
            target_msg = await channel.fetch_message(int(m_id))
        except discord.NotFound:
            return ["The specific message in the link could not be found or was deleted."]
        except discord.Forbidden:
            return ["The bot does not have permission to read the specific message in the link."]
        except Exception:
            return ["An error occurred while fetching the specific message."]

        limit = CHANNEL_READ_WINDOW_SIZE
        half = limit // 2
        before_msgs = []
        async for msg in channel.history(limit=half, before=target_msg):
            before_msgs.append(msg)
        before_msgs.reverse()  # newest-first -> chronological
        after_msgs = []
        async for msg in channel.history(limit=half, after=target_msg):
            after_msgs.append(msg)
        window = before_msgs + [target_msg] + after_msgs
        parts = await _build_window_parts(window, target_msg.id, "target")
        header = (f"Transcript of #{channel.name} around the linked message "
                  f"({len(window)} messages, oldest first):")
        return [header] + parts

    channel_input = channel_input.strip()
    channel = None
    mention_match = re.match(r"^<#(\d+)>$", channel_input)
    if mention_match:
        channel = bot.get_channel(int(mention_match.group(1)))
    elif channel_input.isdigit():
        channel = bot.get_channel(int(channel_input))
    else:
        guild = bot.get_guild(guild_id)
        if guild:
            for ch in guild.text_channels:
                if ch.name.lower() == channel_input.lower():
                    channel = ch
                    break
    if channel is None:
        try:
            if channel_input.isdigit() or mention_match:
                cid = int(mention_match.group(1)) if mention_match else int(channel_input)
                channel = await bot.fetch_channel(cid)
        except discord.NotFound:
            return ["The specified channel could not be found."]
        except discord.Forbidden:
            return ["The bot does not have access to the specified channel."]
        except Exception:
            return ["An error occurred while accessing the specified channel."]
    if channel is None or not isinstance(channel, discord.abc.Messageable):
        return ["The specified channel could not be found or is not readable."]

    window = []
    async for msg in channel.history(limit=CHANNEL_READ_WINDOW_SIZE):
        window.append(msg)
    window.reverse()  # history returns newest-first; reverse to chronological
    if not window:
        return [f"The channel #{getattr(channel, 'name', 'unknown')} has no readable messages."]
    parts = await _build_window_parts(window, None, "window")
    header = f"Transcript of #{channel.name} ({len(window)} messages, oldest first):"
    return [header] + parts