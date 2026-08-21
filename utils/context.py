import discord
from utils.formatting import clean_answer_text


def build_server_channel_list(guild: discord.Guild) -> str:
    if guild is None: return "(Not in a server)"
    channels = []
    for ch in guild.text_channels:
        perms = ch.permissions_for(guild.me)
        if perms.view_channel and perms.read_messages:
            topic = f" — {ch.topic}" if ch.topic else ""
            channels.append(f"#{ch.name}{topic}")
    if not channels: return "(No accessible channels)"
    return "\n".join(f"- {c}" for c in channels)


async def get_conversation_context(message: discord.Message, bot_user_id: int, bot_display_name: str,
                                   max_history: int = 15) -> list[dict]:
    history_msgs = []
    if message.reference and message.reference.message_id:
        current_ref = message.reference
        chain = []
        while current_ref and len(chain) < max_history:
            try:
                ref_msg = current_ref.resolved if current_ref.resolved else await message.channel.fetch_message(
                    current_ref.message_id)
                if not ref_msg: break
                chain.append(ref_msg)
                current_ref = ref_msg.reference if ref_msg.reference and ref_msg.reference.message_id else None
            except Exception:
                break
        chain.reverse()
        history_msgs = chain
    else:
        async for msg in message.channel.history(limit=max_history + 10, before=message):
            if msg.author.bot and msg.author.id != bot_user_id: continue
            if not msg.clean_content and not msg.attachments: continue
            history_msgs.append(msg)
            if len(history_msgs) >= max_history: break
        history_msgs.reverse()

    context = []
    for msg in history_msgs:
        if msg.id == message.id: continue
        content = msg.clean_content
        if not content and msg.attachments:
            has_image = any(att.content_type and att.content_type.startswith("image/") for att in msg.attachments)
            content = "[User attached an image]" if has_image else "[User attached a file]"
        if not content: continue
        if msg.author.id == bot_user_id:
            context.append({"role": "assistant", "content": clean_answer_text(content)})
        else:
            context.append({"role": "user", "content": f"{msg.author.display_name}: {content}"})
    return context