import re
import logging
import discord
from config import CHANNEL_READ_WINDOW_SIZE

log = logging.getLogger("rag-bot")

DISCORD_MSG_LINK_RE = re.compile(r"^https?://(?:discord\.com|discordapp\.com)/channels/(\d+|@me)/(\d+)/(\d+)$")
_MAX_OUTPUT_CHARS = 10000
_MAX_SINGLE_MSG_CHARS = 1000


def _format_window(window: list[discord.Message], target_id: int | None) -> str:
    lines = []
    for msg in window:
        if msg.id == target_id:
            lines.append(">>> TARGET MESSAGE (linked by user) <<<")
        ts = msg.created_at.strftime("%Y-%m-%d %H:%M UTC")
        content = msg.clean_content or "(no text content)"
        if len(content) > _MAX_SINGLE_MSG_CHARS:
            content = content[:_MAX_SINGLE_MSG_CHARS] + "... [truncated]"

        author = msg.author
        if isinstance(author, discord.Member) and author.nick:
            display_name = author.nick
        elif getattr(author, "global_name", None):
            display_name = author.global_name
        else:
            display_name = author.name

        author_label = f"{display_name} (@{author.name})"

        lines.append(f"[{ts}] {author_label}: {content}")

    result = "\n".join(lines)
    if len(result) > _MAX_OUTPUT_CHARS:
        result = result[:_MAX_OUTPUT_CHARS] + "\n\n[Output truncated due to length.]"
    return result


async def read_channel_content(bot: discord.Client, guild_id: int, channel_input: str, message_link: str) -> str:
    if message_link:
        match = DISCORD_MSG_LINK_RE.match(message_link.strip())
        if not match:
            return "The provided message link is invalid or not recognized."

        g_id, c_id, m_id = match.groups()
        if g_id == "@me":
            return "The provided link is a direct message link, which is invalid for this tool."
        if int(g_id) != guild_id:
            return "The provided link points to a different server, which is invalid for this tool."

        channel = bot.get_channel(int(c_id))
        if channel is None:
            try:
                channel = await bot.fetch_channel(int(c_id))
            except discord.NotFound:
                return "The channel in the provided link could not be found."
            except discord.Forbidden:
                return "The bot does not have access to the channel in the provided link."
            except Exception:
                return "An error occurred while accessing the channel in the provided link."

        if not isinstance(channel, discord.abc.Messageable):
            return "The provided link points to a channel type that cannot be read."

        try:
            target_msg = await channel.fetch_message(int(m_id))
        except discord.NotFound:
            return "The specific message in the link could not be found or was deleted."
        except discord.Forbidden:
            return "The bot does not have permission to read the specific message in the link."
        except Exception:
            return "An error occurred while fetching the specific message."

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
        return _format_window(window, target_msg.id)

    else:
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
                return "The specified channel could not be found."
            except discord.Forbidden:
                return "The bot does not have access to the specified channel."
            except Exception:
                return "An error occurred while accessing the specified channel."

        if channel is None or not isinstance(channel, discord.abc.Messageable):
            return "The specified channel could not be found or is not readable."
        limit = CHANNEL_READ_WINDOW_SIZE
        window = []
        async for msg in channel.history(limit=limit):
            window.append(msg)
        window.reverse()  # history returns newest-first; reverse to chronological
        if not window:
            return f"The channel #{getattr(channel, 'name', 'unknown')} has no readable messages."

        return _format_window(window, target_id=None)