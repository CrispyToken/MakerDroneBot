import re
import logging
import aiohttp
from pathlib import Path
from typing import Any
from pydantic_ai import Agent, RunContext, ImageUrl
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.capabilities import WebFetch
from pydantic_ai_harness import (
    OutputGuardrail, GuardrailResult,
    PromptInjectionDefender, CodeMode
)
from pydantic_ai_skills import SkillsCapability, GitSkillsRegistry
from pydantic_monty import MountDir
from pydantic_ai_harness.repair_tool_arguments import RepairToolArguments

from openai import AsyncOpenAI
from config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL_ID, SKILLS_DIRS, LLM_SERVER_MANAGER, LLM_CONTEXT_WINDOW
from core.memory import memory_remember, memory_recall_dynamic, memory_recall_knowledge
from core.prompt_builder import BotDependencies, build_system_prompt
from services.web_search import execute_web_search
from services.game_db import game_database
from utils.formatting import sanitize_bot_style

log = logging.getLogger("rag-bot")

llm_provider = OpenAIProvider(
    openai_client=AsyncOpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY or "no-key",
        timeout=9000.0,
    )
)

def enforce_bot_style(output: str) -> GuardrailResult:
    if not isinstance(output, str):
        return GuardrailResult.allow()
    cleaned = sanitize_bot_style(output)
    if cleaned != output:
        return GuardrailResult.replace(cleaned)
    return GuardrailResult.allow()

_SANDBOX_SCRATCHPAD = Path("data/sandbox_scratchpad").resolve()
_SANDBOX_SCRATCHPAD.mkdir(parents=True, exist_ok=True)

_existing_skills_dirs = [str(d) for d in SKILLS_DIRS if d.is_dir()]
_capabilities = [
    OutputGuardrail(guard=enforce_bot_style),
    WebFetch(local=True),
    PromptInjectionDefender(),
    RepairToolArguments(),
    CodeMode(
        tools=[],
        mount=MountDir(host_path=str(_SANDBOX_SCRATCHPAD), virtual_path='/scratchpad', mode='read-write')
    )
]
_registries = [
    # Dynamically pull Anthropic's official skills at runtime
    # GitSkillsRegistry('https://github.com/anthropics/skills', path='skills')
]

if _existing_skills_dirs:
    # Combine local directories with remote registries
    _capabilities.append(SkillsCapability(_existing_skills_dirs, registries=_registries))
else:
    # If no local dirs exist, use the remote registries
    _capabilities.append(SkillsCapability(registries=_registries))


async def _get_server_context_size() -> int:
    if LLM_SERVER_MANAGER != "llamacpp":
        return LLM_CONTEXT_WINDOW

    base = LLM_BASE_URL.rsplit("/v1", 1)[0]
    url = f"{base}/slots"
    headers = {"Authorization": f"Bearer {LLM_API_KEY}"} if LLM_API_KEY else {}
    try:
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=headers) as resp:
                if resp.status == 200:
                    slots = await resp.json()
                    if slots and isinstance(slots, list):
                        n_ctx = slots[0].get("n_ctx")
                        if isinstance(n_ctx, int) and n_ctx > 0:
                            return n_ctx
    except Exception:
        log.warning("Could not query server context size; using fallback %d.", LLM_CONTEXT_WINDOW)
    return LLM_CONTEXT_WINDOW


async def get_agent() -> Agent[BotDependencies, str]:
    ctx_size = await _get_server_context_size()
    model = OpenAIChatModel(
        LLM_MODEL_ID,
        provider=llm_provider,
        profile=OpenAIModelProfile(
            openai_chat_supports_multiple_system_messages=False,
            context_window=ctx_size,
        ),
        settings={"max_tokens": ctx_size // 4}
    )
    agent = Agent(
        model=model,
        deps_type=BotDependencies,
        output_type=str,
        capabilities=_capabilities,
        retries=3
    )

    @agent.instructions
    def dynamic_system_prompt(ctx: RunContext[BotDependencies]) -> str:
        return build_system_prompt(ctx.deps)

    @agent.tool
    async def search_game_knowledge(ctx: RunContext[BotDependencies], query: str) -> str:
        """
        Searches the static knowledge base for game lore, rules, mechanics, ship modules, and
        server-specific information about Event Horizon. Use when the user asks about the game,
        its mechanics, ships, modules, or server rules. Do NOT use for general knowledge, recent
        events, or things that happened in chat.
        """
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
        """
        Searches your dynamic memory for facts, decisions, and events learned from past
        conversations. Use when the user asks about something that was discussed previously, a
        decision that was made, or context from earlier interactions. Do NOT use for game lore
        or static knowledge.
        """
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
        """
        Saves a new, important fact to your long-term dynamic memory. Use when a user tells you
        something new about the game, server rules, or community that you didn't already know,
        when a staff member gives you a new instruction or rule to remember, or when you observe
        a decision being made that should be recorded. Do NOT use for trivial conversation,
        greetings, or things you already know. Only save facts that would be useful to remember
        in future conversations.
        """
        try:
            await memory_remember(fact)
            return f"Successfully saved dynamic memory: {fact}"
        except Exception as e:
            log.exception("Memory save failed")
            return f"Failed to save memory: {e}"

    @agent.tool
    async def search_web(ctx: RunContext[BotDependencies], query: str) -> str:
        """
        Searches the web for recent information, patch notes, news, or topics not covered in the
        local knowledge base. Use when the user asks about recent events, recent patch notes, or
        things outside your static knowledge. When in doubt about whether you have current or
        complete information, search the web rather than guessing or declining to answer. Prefer
        web search over saying "I don't know" when the information could plausibly be found online.
        """
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
        - message_link: Full Discord message URL, or a bare message ID. The `id:` values shown
          in conversation context and channel transcripts are message IDs and can be passed here
          as-is; a bare ID is resolved as a message in the channel this conversation is taking
          place in. The tool will automatically fetch context around that message.
        Provide either channel or message_link. If message_link is provided, it takes precedence.
        """
        if not ctx.deps.is_staff:
            return "Access denied: This tool can only be used by confirmed staff members."
        message_link = message_link.strip()
        if message_link.startswith("id:"):
            message_link = message_link[3:].strip()
        if message_link.isdigit():
            message_link = f"https://discord.com/channels/{ctx.deps.guild_id}/{ctx.deps.channel_id}/{message_link}"
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