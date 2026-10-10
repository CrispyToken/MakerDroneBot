import logging
from dataclasses import dataclass
from config import SYSTEM_PROMPT_FILE
import discord

log = logging.getLogger("rag-bot")

@dataclass
class BotDependencies:
    bot: discord.Client
    guild_id: int
    channel_id: int
    current_time_str: str
    channel_name: str
    channel_topic: str
    server_channels: str
    is_staff: bool
    staff_role_ids: set[int]
    user_profile: dict | None


def load_system_prompt() -> str:
    try:
        if SYSTEM_PROMPT_FILE.exists():
            return SYSTEM_PROMPT_FILE.read_text(encoding="utf-8").strip()
        log.warning("System prompt file not found at %s. Using basic fallback.", SYSTEM_PROMPT_FILE)
        return "You are a helpful assistant."
    except Exception:
        log.exception("Failed to read system prompt file.")
        return "You are a helpful assistant."


def _apply_placeholders(prompt: str, deps: BotDependencies) -> str:
    if "{{CURRENT_DATE}}" in prompt:
        prompt = prompt.replace("{{CURRENT_DATE}}", deps.current_time_str)
    else:
        prompt = f"{deps.current_time_str}\n\n{prompt}"
    if "{{CURRENT_CHANNEL_NAME}}" in prompt:
        prompt = prompt.replace("{{CURRENT_CHANNEL_NAME}}", deps.channel_name)
    if "{{CURRENT_CHANNEL_TOPIC}}" in prompt:
        topic_text = deps.channel_topic if deps.channel_topic else "(No topic set)"
        prompt = prompt.replace("{{CURRENT_CHANNEL_TOPIC}}", topic_text)
    return prompt


def _channel_list_section(deps: BotDependencies) -> str:
    return (
        "\n\n[Server Channels. For your internal orientation only. "
        "Do not mention this list to users, do not reference it in responses, "
        "and do not suggest channels unless directly asked.]\n"
        f"{deps.server_channels}"
    )

def build_system_prompt(deps: BotDependencies) -> str:
    prompt = _apply_placeholders(load_system_prompt(), deps)
    prompt += _channel_list_section(deps)
    return prompt