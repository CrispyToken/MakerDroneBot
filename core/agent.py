from dataclasses import dataclass, field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
import cognee
import logging
from config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL_ID, SYSTEM_PROMPT_FILE
from core.memory import cognee_in_background, cognee_recall, release_cognee_lock
from services.web_search import execute_web_search
import services.skills as skills_module

log = logging.getLogger("rag-bot")

llm_provider = OpenAIProvider(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY or "no-key"
)


@dataclass
class BotDependencies:
    user_profile: dict | None
    current_time_str: str
    channel_name: str
    channel_topic: str
    server_channels: str
    active_skills: list = field(default_factory=list)


def load_system_prompt() -> str:
    try:
        if SYSTEM_PROMPT_FILE.exists():
            return SYSTEM_PROMPT_FILE.read_text(encoding="utf-8").strip()
        else:
            log.warning("System prompt file not found at %s. Using basic fallback.", SYSTEM_PROMPT_FILE)
            return "You are a helpful assistant."
    except Exception as e:
        log.exception("Failed to read system prompt file.")
        return "You are a helpful assistant."


def build_system_prompt(ctx: RunContext[BotDependencies]) -> str:
    deps = ctx.deps
    base_prompt = load_system_prompt()
    if "{{CURRENT_TIME}}" in base_prompt:
        base_prompt = base_prompt.replace("{{CURRENT_TIME}}", deps.current_time_str)
    else:
        base_prompt = f"{deps.current_time_str}\n\n{base_prompt}"
    if "{{CURRENT_CHANNEL_NAME}}" in base_prompt:
        base_prompt = base_prompt.replace("{{CURRENT_CHANNEL_NAME}}", deps.channel_name)
    if "{{CURRENT_CHANNEL_TOPIC}}" in base_prompt:
        topic_text = deps.channel_topic if deps.channel_topic else "(No topic set)"
        base_prompt = base_prompt.replace("{{CURRENT_CHANNEL_TOPIC}}", topic_text)

    channel_list_section = (
        "\n\n[Server Channels — for your internal orientation only. "
        "Do not mention this list to users, do not reference it in responses, "
        "and do not suggest channels unless directly asked.]\n"
        f"{deps.server_channels}"
    )
    base_prompt += channel_list_section

    if deps.user_profile:
        join_str = deps.user_profile['join_date'][:10] if deps.user_profile['join_date'] else 'Unknown'
        user_context = (
            f"\n\n[User Profile Context]\n"
            f"You are currently speaking to {deps.user_profile['display_name']} (@{deps.user_profile['username']}).\n"
            f"Server Roles: {deps.user_profile['roles'] or 'None'}\n"
            f"Member since: {join_str}\n"
            f"Total messages sent in server: {deps.user_profile['message_count']}\n"
        )
        if deps.user_profile['staff_notes']:
            user_context += f"Staff Notes regarding this user: {deps.user_profile['staff_notes']}\n"
        base_prompt += user_context

    if skills_module.skill_manager:
        skills_context = skills_module.skill_manager.get_skills_prompt()
        if skills_context:
            base_prompt += f"\n\n{skills_context}"

    if deps.active_skills:
        for skill in deps.active_skills:
            base_prompt += (
                f"\n\n[ACTIVE SKILL: {skill.name}]\n"
                "This skill was explicitly activated for the current task. Its instructions "
                "are MANDATORY and take precedence over your defaults. Follow them exactly, "
                "from the very beginning of your work.\n"
                f"--- BEGIN SKILL {skill.name} ---\n{skill.content}\n--- END SKILL {skill.name} ---"
            )
    return base_prompt


def get_agent() -> Agent[BotDependencies, str]:
    model = OpenAIChatModel(LLM_MODEL_ID, provider=llm_provider)
    agent = Agent(
        model=model,
        deps_type=BotDependencies,
        output_type=str,
    )

    @agent.system_prompt
    def dynamic_system_prompt(ctx: RunContext[BotDependencies]) -> str:
        return build_system_prompt(ctx)

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
            f"for this task — follow them exactly:\n\n{skill.content}"
        )

    @agent.tool
    async def search_game_knowledge(ctx: RunContext[BotDependencies], query: str) -> str:
        try:
            insights = await cognee_recall(query)
            if not insights: return "No relevant game knowledge found."
            formatted = []
            seen = set()
            for insight in insights:
                ds_name = getattr(insight, "dataset_name", None) or (
                    insight.get("dataset_name") if isinstance(insight, dict) else None)
                text = getattr(insight, "text", None) or (
                    insight.get("text", str(insight)) if isinstance(insight, dict) else str(insight))
                if ds_name == "event_horizon_dynamic": continue
                if text and text not in seen:
                    seen.add(text)
                    formatted.append(text)
            if not formatted: return "No relevant game knowledge found."
            return "\n".join(f"{i}. {t}" for i, t in enumerate(formatted[:5], 1))
        except Exception as e:
            log.exception("Cognee game knowledge recall failed")
            return f"Game knowledge search failed: {e}"

    @agent.tool
    async def search_memory(ctx: RunContext[BotDependencies], query: str) -> str:
        try:
            insights = await cognee_recall(query)
            if not insights: return "No relevant memories found."
            formatted = []
            seen = set()
            for insight in insights:
                ds_name = getattr(insight, "dataset_name", None) or (
                    insight.get("dataset_name") if isinstance(insight, dict) else None)
                text = getattr(insight, "text", None) or (
                    insight.get("text", str(insight)) if isinstance(insight, dict) else str(insight))
                if ds_name != "event_horizon_dynamic": continue
                if text and text not in seen:
                    seen.add(text)
                    formatted.append(text)
            if not formatted: return "No relevant memories found."
            return "\n".join(f"{i}. {t}" for i, t in enumerate(formatted[:5], 1))
        except Exception as e:
            log.exception("Cognee memory recall failed")
            return f"Memory search failed: {e}"

    @agent.tool
    async def save_memory(ctx: RunContext[BotDependencies], fact: str) -> str:
        try:
            await cognee_in_background(cognee.remember, fact, dataset_name="event_horizon_dynamic")
            return f"Successfully saved and processed dynamic memory: {fact}"
        except Exception as e:
            log.exception("Cognee remember failed")
            return f"Failed to save memory: {e}"
        finally:
            await release_cognee_lock()

    @agent.tool
    async def search_web(ctx: RunContext[BotDependencies], query: str) -> str:
        return await execute_web_search(query)

    return agent