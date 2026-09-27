import re
import discord
from io import BytesIO
from config import DISCORD_CHAR_LIMIT, DISCORD_MAX_SPLIT_CHARS

_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF"
    "\U0001FA70-\U0001FAFF"
    "\U00002600-\U000026FF"
    "]+", flags=re.UNICODE)

def sanitize_bot_style(text: str) -> str:
    if not text:
        return text
    text = text.replace('\u2014', '-')
    text = _EMOJI_RE.sub('', text)
    return text

def strip_thinking_tokens(text: str) -> str:
    text = text or ""
    text = re.sub(r"(?is)<think>.*?</think>", "", text)
    text = re.sub(r"(?im)^\s*<think>\s*$", "", text)
    text = re.sub(r"(?im)^\s*</think>\s*$", "", text)
    return text.strip()

def clean_answer_text(text: str) -> str:
    text = strip_thinking_tokens(text)
    text = re.sub(r"(?im)^\s*(sources?|citations?|references?)\s*:.*$", "", text)
    text = re.sub(r"(?is)\n+\*\*Sources:\*\*.*$", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

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
    answer = clean_answer_text(answer)
    if not answer:
        await message.reply("I could not generate a sendable response.")
        return
    length = len(answer)
    if length <= DISCORD_CHAR_LIMIT:
        await message.reply(answer)
        return
    if length <= DISCORD_MAX_SPLIT_CHARS:
        parts = split_for_discord(answer, limit=DISCORD_CHAR_LIMIT)
        for i, part in enumerate(parts, 1):
            await message.reply(f"Part {i}/{len(parts)}\n{part}")
        return
    file = discord.File(BytesIO(answer.encode("utf-8")), filename="response.txt")
    try:
        await message.reply(
            f"The response was {length} characters, which is too long to send directly. "
            "The full response is attached as a text file.",
            file=file,
        )
    except discord.HTTPException:
        await message.reply("The response was too long to send, and the attachment could not be delivered.")