import re
import logging
from typing import Any
from pydantic_ai import Agent, RunContext, ImageUrl
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.capabilities import WebFetch
from pydantic_ai_harness import (
    SystemReminders, OutputGuardrail, GuardrailResult,
    PromptInjectionDefender
)
from pydantic_ai_skills import SkillsCapability
from pydantic_ai_harness.system_reminders import GoalReanchor
from pydantic_ai_harness.repair_tool_arguments import RepairToolArguments

from config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL_ID, SKILLS_DIRS
from core.memory import memory_remember, memory_recall_dynamic, memory_recall_knowledge
from core.prompt_builder import BotDependencies, build_system_prompt
from services.web_search import execute_web_search
from services.game_db import game_database

log = logging.getLogger("rag-bot")

llm_provider = OpenAIProvider(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY or "no-key"
)

_EMOJI_RE = re.compile("["
                       u"\U0001F600-\U0001F64F"
                       u"\U0001F300-\U0001F5FF"
                       u"\U0001F680-\U0001F6FF"
                       u"\U0001F1E0-\U0001F1FF"
                       u"\U00002702-\U000027B0"
                       u"\U000024C2-\U0001F251"
                       u"\U0001F900-\U0001F9FF"
                       u"\U0001FA70-\U0001FAFF"
                       u"\U00002600-\U000026FF"
                       "]+", flags=re.UNICODE)


def enforce_bot_style(output: str) -> GuardrailResult:
    if not isinstance(output, str):
        return GuardrailResult.allow()

    if '—' in output:
        return GuardrailResult.retry(
            "You used an EM dash (—). This is strictly forbidden. Do not use em dashes, "
            "and do not substitute them with regular dashes. Rewrite your response without them."
        )

    if _EMOJI_RE.search(output):
        return GuardrailResult.retry(
            "You used an emoji. This is strictly forbidden. Rewrite your response without any emojis."
        )

    return GuardrailResult.allow()


_existing_skills_dirs = [str(d) for d in SKILLS_DIRS if d.is_dir()]
_capabilities = [
    SystemReminders(dynamic_reminders=[GoalReanchor()]),
    OutputGuardrail(guard=enforce_bot_style),
    WebFetch(local=True),
    PromptInjectionDefender(block_high_risk=True),
    RepairToolArguments()
]
if _existing_skills_dirs:
    _capabilities.append(SkillsCapability(_existing_skills_dirs))


def get_agent() -> Agent[BotDependencies, str]:
    model = OpenAIChatModel(LLM_MODEL_ID, provider=llm_provider)
    agent = Agent(
        model=model,
        deps_type=BotDependencies,
        output_type=str,
        capabilities=_capabilities
    )

    @agent.system_prompt
    def dynamic_system_prompt(ctx: RunContext[BotDependencies]) -> str:
        return build_system_prompt(ctx.deps)

    @agent.tool
    async def search_game_knowledge(ctx: RunContext[BotDependencies], query: str) -> str:
        try:
            context = await memory_recall_knowledge(query)
            if not context or not context.strip():
                return "No relevant game knowledge found."
            return context.strip()
        except Exception as e:
            log.exception("Game knowledge recall failed")
            return f"Game knowledge search failed: {e}"

    @agent.tool
    async def search_memory(ctx: RunContext[BotDependencies], query: str) -> str:
        try:
            context = await memory_recall_dynamic(query)
            if not context or not context.strip():
                return "No relevant memories found."
            return context.strip()
        except Exception as e:
            log.exception("Dynamic memory recall failed")
            return f"Memory search failed: {e}"

    @agent.tool
    async def save_memory(ctx: RunContext[BotDependencies], fact: str) -> str:
        try:
            await memory_remember(fact)
            return f"Successfully saved dynamic memory: {fact}"
        except Exception as e:
            log.exception("Memory save failed")
            return f"Failed to save memory: {e}"

    @agent.tool
    async def search_web(ctx: RunContext[BotDependencies], query: str) -> str:
        return await execute_web_search(query)

    @agent.tool
    async def game_lookup(ctx: RunContext[BotDependencies], query: str) -> str:
        """
        Looks up verified Event Horizon game data by name or keyword. Use this for any question
        about concrete game facts: ships, modules, weapons, ammunition, component stats, costs,
        workshop levels, factions, or ship builds. Pass the entity name as the user said it
        (e.g. "Valkyrie", "heavy railgun"); the tool works out whether it is a ship or a module
        and returns verified database values for every match. Quote those values exactly and
        never invent numbers. If multiple matches come back, present the candidates or ask
        which one is meant.
        """
        return game_database.lookup(query)

    @agent.tool
    async def read_channel(ctx: RunContext[BotDependencies], channel: str = "", message_link: str = "") -> Any:
        """
        Reads messages from a Discord channel, including image attachments. Use this when a user
        asks to see what's happening in a specific channel, or provides a link to a specific
        message. For a channel read, every image attached to messages in the read window is
        included; for a message link, only the linked message's own attachments are included.
        RESTRICTION: Only use this tool if the [Request Authority] block in your system prompt
        confirms the requesting user is CONFIRMED STAFF. Do not use this tool for regular members.
        Parameters:
        - channel: Channel name, mention (e.g., <#123456>), or ID. Provide this when the user
          asks to read a channel.
        - message_link: Full Discord message URL. Provide this when the user shares a link to a
          specific message. The tool will automatically fetch context around that message.
        Provide either channel or message_link. If message_link is provided, it takes precedence.
        """
        if not ctx.deps.is_staff:
            return "Access denied: This tool can only be used by confirmed staff members."
        from services.channel_reader import read_channel_content
        parts = await read_channel_content(ctx.deps.bot, ctx.deps.guild_id, channel, message_link)
        content: list[str | ImageUrl] = []
        for part in parts:
            if isinstance(part, str):
                content.append(part)
            else:
                content.append(ImageUrl(url=part["image_url"]["url"]))
        if len(content) == 1 and isinstance(content[0], str):
            return content[0]
        return content

    return agent