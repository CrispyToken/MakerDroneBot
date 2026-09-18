import asyncio
import logging
import discord
from datetime import datetime
from core.agent import get_agent, BotDependencies
from core.db import get_user_profile
from utils.context import build_server_channel_list, get_conversation_context
from utils.attachments import collect_image_attachments, collect_text_attachments
from utils.formatting import send_final_answer
import re
import services.skills as skills_module
from core.llm_reasoning import chat_reasoning
from core.console import print_user_line
from core.locks import llm_lock, track_llm_task
from config import CONVERSATION_MAX_HISTORY

log = logging.getLogger("rag-bot")

def detect_explicit_skills(question: str) -> list:
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

async def _keep_typing(channel: discord.abc.Messageable, stop_event: asyncio.Event):
    """Send a typing ping every 7 seconds until the stop event is set."""
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


async def answer_question(bot: discord.Client, message: discord.Message, question: str):
    if llm_lock.locked():
        log.info("Chat request rejected: LLM is busy with another task.")
        await message.reply("I'm currently processing another task. Please try again in a minute.")
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

        now = datetime.now().astimezone()
        tz_name = now.tzname() or "Local Time"
        tz_offset = now.strftime("%z")
        formatted_offset = f"{tz_offset[:3]}:{tz_offset[3:]}" if len(tz_offset) == 5 else tz_offset
        current_time_str = (
            f"Current Date and Time: {now.strftime('%A, %B %d, %Y at %H:%M:%S')} "
            f"{tz_name} (UTC{formatted_offset})"
        )

        profile = await get_user_profile(message.author.id)
        channel_name = message.channel.name if message.guild else "Direct Message"
        channel_topic = getattr(message.channel, 'topic', None) or ""
        server_channels = build_server_channel_list(message.guild)

        deps = BotDependencies(
            user_profile=profile,
            current_time_str=current_time_str,
            channel_name=channel_name,
            channel_topic=channel_topic,
            server_channels=server_channels,
            active_skills=active_skills,
        )

        prompt_text = question_for_model
        if text_blocks:
            prompt_text += "\n\n" + "\n\n".join(text_blocks)
        if message.author.id != bot.user.id:
            prompt_text = f"{message.author.display_name}: " + prompt_text

        conversation_history = await get_conversation_context(message, bot.user.id, bot.user.display_name,
                                                              max_history=CONVERSATION_MAX_HISTORY)
        history_text = ""
        if conversation_history:
            history_text = (
                    "[Recent Conversation History — chronological, oldest to newest, timestamps UTC]\n"
                    + "\n".join(conversation_history)
                    + "\n[End of History]\n\n"
            )

        if images:
            from pydantic_ai import ImageUrl
            pydantic_images = [ImageUrl(url=img["image_url"]["url"]) for img in images]
            full_prompt = prompt_text
            if history_text:
                full_prompt = history_text + full_prompt
            user_content = [full_prompt, *pydantic_images]
        else:
            user_content = history_text + prompt_text

        agent = get_agent()

        if isinstance(user_content, str):
            print_user_line(user_content)
        else:
            print_user_line(next((c for c in user_content if isinstance(c, str)), "(multimodal input)"))

        try:
            await asyncio.wait_for(llm_lock.acquire(), timeout=5)
        except asyncio.TimeoutError:
            await message.reply("I'm currently processing another task. Please try again in a minute.")
            return
        try:
            async with track_llm_task("chat"):
                async with chat_reasoning():
                    result = await agent.run(user_content, deps=deps)
        except asyncio.CancelledError:
            log.info("Chat turn interrupted by staff command.")
            return
        finally:
            llm_lock.release()
        await send_final_answer(message, result.output)

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