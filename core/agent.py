import logging
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL_ID
from core.memory import memory_remember, memory_recall_dynamic, memory_recall_knowledge
from core.prompt_builder import BotDependencies, build_system_prompt
from services.web_search import execute_web_search
import services.skills as skills_module
from services.game_db import game_database

log = logging.getLogger("rag-bot")

llm_provider = OpenAIProvider(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY or "no-key"
)


def get_agent() -> Agent[BotDependencies, str]:
    model = OpenAIChatModel(LLM_MODEL_ID, provider=llm_provider)
    agent = Agent(
        model=model,
        deps_type=BotDependencies,
        output_type=str,
    )

    @agent.system_prompt
    def dynamic_system_prompt(ctx: RunContext[BotDependencies]) -> str:
        return build_system_prompt(ctx.deps)

    @agent.tool
    async def activate_skill(ctx: RunContext[BotDependencies], skill_name: str) -> str:
        """
        Activates a specific Agent Skill. Use this when the user explicitly asks to use a skill,
        or when a complex task perfectly matches a skill's description.
        """
        manager = skills_module.skill_manager
        if not manager:
            return "Skill manager not initialized."

        skill = manager.get_skill(skill_name)
        if not skill:
            available = ", ".join(manager.skills.keys()) if manager.skills else "None"
            return f"Skill '{skill_name}' not found. Available skills: {available}"

        if skill not in ctx.deps.active_skills:
            ctx.deps.active_skills.append(skill)

        return (
            f"Skill '{skill.name}' is now ACTIVE. The instructions below are MANDATORY "
            f"for this task. Follow them exactly:\n\n{skill.content}"
        )

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
    async def read_channel(ctx: RunContext[BotDependencies], channel: str = "", message_link: str = "") -> str:
        """
        Reads messages from a Discord channel. Use this when a user asks to see what's happening
        in a specific channel, or provides a link to a specific message.

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
        return await read_channel_content(ctx.deps.bot, ctx.deps.guild_id, channel, message_link)

    return agent