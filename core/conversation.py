import asyncio
import logging
import discord
from datetime import datetime
from core.agent import get_agent, BotDependencies
from core.db import get_user_profile, get_staff_role_ids
from utils.context import build_server_channel_list, get_conversation_context
from utils.attachments import collect_image_attachments, collect_text_attachments
from utils.formatting import send_final_answer
import re
from services.skills import Skill
import services.skills as skills_module
from core.llm_reasoning import chat_reasoning
from core.console import print_user_line
from core.locks import llm_lock, track_llm_task, active_llm_task_label
from config import CONVERSATION_MAX_HISTORY

log = logging.getLogger("rag-bot")

def detect_explicit_skills(question: str) -> list[Skill]:
    manager = skills_module.skill_manager
    if not manager or not manager.skills:
        return []
    q = question.lower()
    found = []
    for name, skill in manager.skills.items():
        patterns = [
            rf'\b(?:use|using|activate|apply|employ|run)\s+(?:the\s+)?{re.escape(name)}(?:\s+skill)?\b',
            rf'\bwith\s+(?:the\s+)?{re.escape(name)}\s+skill\b',
        ]
        if any(re.search(p, q) for p in patterns):
            found.append(skill)
    return found

def _busy_message(busy_label: str | None) -> str:
    if busy_label:
        return f"I'm currently processing another task: {busy_label}. Please try again in a minute."
    return "I'm currently processing another task. Please try again in a minute."

async def _keep_typing(channel: discord.abc.Messageable, stop_event: asyncio.Event) -> None:
    while not stop_event.is_set():
        try:
            await channel.typing()
        except discord.HTTPException:
            pass
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=7.0)
            break
        except asyncio.TimeoutError:
            continue

def _format_current_time() -> str:
    now = datetime.now().astimezone()
    tz_name = now.tzname() or "Local Time"
    tz_offset = now.strftime("%z")
    formatted_offset = f"{tz_offset[:3]}:{tz_offset[3:]}" if len(tz_offset) == 5 else tz_offset
    return (
        f"Current Date and Time: {now.strftime('%A, %B %d, %Y at %H:%M:%S')} "
        f"{tz_name} (UTC{formatted_offset})"
    )

async def _gather_dependencies(message: discord.Message, active_skills: list[Skill]) -> BotDependencies:
    profile = await get_user_profile(message.author.id)
    channel_name = message.channel.name if message.guild else "Direct Message"
    channel_topic = getattr(message.channel, 'topic', None) or ""
    server_channels = build_server_channel_list(message.guild)
    user_roles: list[str] | None = None
    is_staff = False
    if isinstance(message.author, discord.Member):
        user_roles = [role.name for role in message.author.roles if role.name != "@everyone"]
        staff_role_ids = await get_staff_role_ids()
        is_staff = bool({role.id for role in message.author.roles} & staff_role_ids)
    return BotDependencies(
        user_profile=profile,
        current_time_str=_format_current_time(),
        channel_name=channel_name,
        channel_topic=channel_topic,
        server_channels=server_channels,
        user_roles=user_roles,
        is_staff=is_staff,
        active_skills=active_skills,
    )

async def _build_prompt_text(message: discord.Message, bot_user_id: int, question: str, text_blocks: list[str]) -> str:
    prompt_text = question
    if text_blocks:
        prompt_text += "\n\n" + "\n\n".join(text_blocks)
    if message.author.id != bot_user_id:
        prompt_text = f"{message.author.display_name}: " + prompt_text
    return prompt_text

async def _assemble_user_content(message: discord.Message, bot_user_id: int, bot_display_name: str,
                                 question: str, text_blocks: list[str], images: list[dict]) -> str | list:
    conversation_history = await get_conversation_context(
        message, bot_user_id, bot_display_name, max_history=CONVERSATION_MAX_HISTORY
    )
    history_text = ""
    if conversation_history:
        history_text = (
            "[Recent Conversation History — chronological, oldest to newest, timestamps UTC]\n"
            + "\n".join(conversation_history)
            + "\n[End of History]\n\n"
        )

    prompt_text = await _build_prompt_text(message, bot_user_id, question, text_blocks)

    if images:
        from pydantic_ai import ImageUrl
        pydantic_images = [ImageUrl(url=img["image_url"]["url"]) for img in images]
        full_prompt = prompt_text
        if history_text:
            full_prompt = history_text + full_prompt
        return [full_prompt, *pydantic_images]
    else:
        return history_text + prompt_text

async def _execute_agent(message: discord.Message, user_content: str | list, deps: BotDependencies) -> str | None:
    agent = get_agent()

    if isinstance(user_content, str):
        print_user_line(user_content)
    else:
        print_user_line(next((c for c in user_content if isinstance(c, str)), "(multimodal input)"))

    try:
        await asyncio.wait_for(llm_lock.acquire(), timeout=5)
    except asyncio.TimeoutError:
        await message.reply(_busy_message(active_llm_task_label()))
        return None

    try:
        async with track_llm_task("chat"):
            async with chat_reasoning():
                result = await agent.run(user_content, deps=deps)
        return result.output
    except asyncio.CancelledError:
        log.info("Chat turn interrupted by staff command.")
        return None
    finally:
        llm_lock.release()

async def answer_question(bot: discord.Client, message: discord.Message, question: str) -> None:
    if llm_lock.locked():
        busy_label = active_llm_task_label()
        log.info("Chat request rejected: LLM is busy with another task (%s).", busy_label or "unknown")
        await message.reply(_busy_message(busy_label))
        return

    stop_typing = asyncio.Event()
    typing_task = asyncio.create_task(_keep_typing(message.channel, stop_typing))

    try:
        images, image_warnings = await collect_image_attachments(message)
        text_blocks, text_warnings = await collect_text_attachments(message)
        for warning in image_warnings + text_warnings:
            log.info("Attachment warning: %s", warning)

        user_question = question.strip()
        active_skills = detect_explicit_skills(user_question)
        if active_skills:
            log.info("Explicit skill activation: %s", ", ".join(s.name for s in active_skills))

        question_for_model = user_question
        if not question_for_model:
            if images:
                question_for_model = "Describe the attached image(s) in detail."
            elif text_blocks:
                question_for_model = "Review the attached text and answer any relevant request."

        if not question_for_model and not images and not text_blocks:
            await message.reply("Provide a question or attach an image/text file.")
            return

        deps = await _gather_dependencies(message, active_skills)
        user_content = await _assemble_user_content(
            message, bot.user.id, bot.user.display_name, question_for_model, text_blocks, images
        )

        output = await _execute_agent(message, user_content, deps)
        if output is not None:
            await send_final_answer(message, output)

    except RuntimeError as e:
        log.exception("Model/request error")
        await message.reply(f"Model/request error:\n`{e}`")
    except Exception as e:
        log.exception("Answer handler failed")
        await message.reply(f"Sorry, something went wrong.\n`{e}`")
    finally:
        stop_typing.set()
        typing_task.cancel()
        try:
            await typing_task
        except asyncio.CancelledError:
            pass