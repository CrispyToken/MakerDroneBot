import re
import discord
from io import BytesIO
from config import DISCORD_CHAR_LIMIT, DISCORD_MAX_SPLIT_CHARS

def serialize_embed(embed: discord.Embed) -> str:
    """Render a Discord embed as plain text so the model can read it."""
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

def split_for_discord(text: str, limit: int = 2000) -> list[str]:
    text = (text or "").strip()
    if not text: return ["(empty response)"]
    parts = []
    while len(text) > limit:
        cut = text.rfind("\n", 0, limit)
        if cut < limit // 2: cut = limit
        parts.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    if text: parts.append(text)
    return parts

async def send_final_answer(message: discord.Message, answer: str):
    answer = (answer or "").strip()
    if not answer:
        await message.reply("I could not generate a sendable response.")
        return
    length = len(answer)
    if length <= DISCORD_CHAR_LIMIT:
        await message.reply(answer)
        return
    if length <= DISCORD_MAX_SPLIT_CHARS:
        for part in split_for_discord(answer, limit=DISCORD_CHAR_LIMIT):
            await message.reply(part)
        return
    file = discord.File(BytesIO(answer.encode("utf-8")), filename="response.md")
    try:
        await message.reply(
            f"The response was {length} characters, which is too long to send directly. "
            "The full response is attached as a text file.",
            file=file,
        )
    except discord.HTTPException:
        await message.reply("The response was too long to send, and the attachment could not be delivered.")