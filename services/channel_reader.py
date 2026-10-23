import re
import logging
import discord
from config import CHANNEL_READ_WINDOW_SIZE
from utils.attachments import collect_image_attachments
from utils.formatting import serialize_embed

log = logging.getLogger("rag-bot")

DISCORD_MSG_LINK_RE = re.compile(r"^https?://(?:discord.com|discordapp.com)/channels/(\d+|@me)/(\d+)/(\d+)$")
def _author_label(msg: discord.Message, is_staff: bool) -> str:
    author = msg.author
    if isinstance(author, discord.Member) and author.nick:
        display_name = author.nick
    elif getattr(author, "global_name", None):
        display_name = author.global_name
    else:
        display_name = author.name
    badge = " [STAFF]" if is_staff else " [MEMBER]"
    return f"{display_name} (@{author.name}){badge}"

def _is_staff(msg: discord.Message, staff_role_ids: set[int]) -> bool:
    if isinstance(msg.author, discord.Member):
        return bool({r.id for r in msg.author.roles} & staff_role_ids)
    return False

def _message_line(msg: discord.Message, target_id: int | None, is_staff: bool) -> str:
    prefix = ">>> TARGET MESSAGE (linked by user) <<<\n" if (target_id is not None and msg.id == target_id) else ""
    ts = msg.created_at.strftime("%Y-%m-%d %H:%M UTC")
    header = f"[{ts} | id:{msg.id}] {_author_label(msg, is_staff)}"
    ref = msg.reference
    if ref and ref.message_id:
        header += f" (in response to id:{ref.message_id})"
    content_parts = []
    if msg.clean_content:
        content_parts.append(msg.clean_content)
    for embed in msg.embeds:
        embed_text = serialize_embed(embed)
        if embed_text:
            content_parts.append(embed_text)
    content = "\n".join(content_parts) if content_parts else "(no text content)"
    return f"{prefix}{header}: {content}"

async def _append_message_parts(parts: list, msg: discord.Message, target_id: int | None,
                                include_images: bool, is_staff: bool) -> None:
    parts.append(_message_line(msg, target_id, is_staff))
    if not include_images or not msg.attachments:
        return
    blocks, warnings = await collect_image_attachments(msg)
    ts = msg.created_at.strftime("%Y-%m-%d %H:%M UTC")
    for warning in warnings:
        parts.append(f"[{ts}] {warning}")
    if blocks:
        parts.extend(blocks)

async def _build_window_parts(window: list[discord.Message], target_id: int | None,
                              images_for: str, staff_role_ids: set[int]) -> list:
    """Assemble transcript parts oldest -> newest. Every message is kept fully
    intact; nothing is truncated or dropped, so the complete context reaches
    the model."""
    parts: list = []
    for msg in window:
        include = images_for == "window" or (images_for == "target" and msg.id == target_id)
        is_staff = _is_staff(msg, staff_role_ids)
        await _append_message_parts(parts, msg, target_id, include, is_staff)
    return parts

async def read_channel_content(bot: discord.Client, guild_id: int, channel_input: str,
                               message_link: str, staff_role_ids: set[int]) -> list:
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
                return ["Missing access to the channel in the provided link."]
            except Exception:
                return ["An error occurred while accessing the channel in the provided link."]
        if not isinstance(channel, discord.abc.Messageable):
            return ["The provided link points to a channel type that cannot be read."]
        try:
            target_msg = await channel.fetch_message(int(m_id))
        except discord.NotFound:
            return ["The specific message in the link could not be found or was deleted."]
        except discord.Forbidden:
            return ["Missing access to read the specific message in the link."]
        except Exception:
            return ["An error occurred while fetching the specific message."]

        limit = CHANNEL_READ_WINDOW_SIZE
        half = limit // 2
        try:
            before_msgs = []
            async for msg in channel.history(limit=half, before=target_msg):
                before_msgs.append(msg)
            before_msgs.reverse()
            after_msgs = []
            async for msg in channel.history(limit=half, after=target_msg):
                after_msgs.append(msg)
        except discord.Forbidden:
            return ["PERMANENT FAILURE: Missing access to read the history around that message. Do not retry. Inform the user."]
        except discord.HTTPException:
            return ["An error occurred while reading the channel history around the linked message. Do not retry."]
        window = before_msgs + [target_msg] + after_msgs
        parts = await _build_window_parts(window, target_msg.id, "target", staff_role_ids)
        header = (f"Transcript of #{channel.name} around the linked message "
                  f"({len(window)} messages, oldest first):")
        return [header] + parts

    channel_input = channel_input.strip()
    if channel_input.startswith("#"):
        channel_input = channel_input[1:]
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
            return [
                "PERMANENT FAILURE: Missing access the specified channel. Do not retry. Inform the user you cannot access it."]
        except Exception:
            return ["An error occurred while accessing the specified channel."]
    if channel is None or not isinstance(channel, discord.abc.Messageable):
        return ["The specified channel could not be found or is not readable. Do not retry. Inform the user."]

    window = []
    try:
        async for msg in channel.history(limit=CHANNEL_READ_WINDOW_SIZE):
            window.append(msg)
    except discord.Forbidden:
        return [
            f"PERMANENT FAILURE: Missing access to read #{channel.name}. Do not retry. Inform the user you cannot access this channel."]
    except discord.HTTPException:
        return [f"An error occurred while reading #{channel.name}. Do not retry."]
    window.reverse()
    if not window:
        return [f"The channel #{getattr(channel, 'name', 'unknown')} has no readable messages."]
    parts = await _build_window_parts(window, None, "window", staff_role_ids)
    header = f"Transcript of #{channel.name} ({len(window)} messages, oldest first):"
    return [header] + parts