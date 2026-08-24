import asyncio
import logging
import discord
from datetime import datetime
from core.agent import get_agent, BotDependencies
from core.db import get_user_profile
from utils.context import build_server_channel_list, get_conversation_context
from utils.attachments import collect_image_attachments, collect_text_attachments
from utils.formatting import send_final_answer
from core.llm_reasoning import chat_reasoning

log = logging.getLogger("rag-bot")


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
    stop_typing = asyncio.Event()
    typing_task = asyncio.create_task(_keep_typing(message.channel, stop_typing))

    try:
        images, image_warnings = await collect_image_attachments(message)
        text_blocks, text_warnings = await collect_text_attachments(message)
        for warning in image_warnings + text_warnings:
            log.info("Attachment warning: %s", warning)

        user_question = question.strip()
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
        )

        prompt_text = question_for_model
        if text_blocks:
            prompt_text += "\n\n" + "\n\n".join(text_blocks)
        if message.author.id != bot.user.id:
            prompt_text = f"{message.author.display_name}: " + prompt_text

        conversation_history = await get_conversation_context(message, bot.user.id, bot.user.display_name,
                                                              max_history=15)
        history_text = ""
        if conversation_history:
            lines = [
                f"{bot.user.display_name}: {msg['content']}" if msg["role"] == "assistant" else msg["content"]
                for msg in conversation_history]
            history_text = "[Recent Conversation History]\n" + "\n".join(lines) + "\n[End of History]\n\n"

        if images:
            user_content = [{"type": "text", "text": prompt_text}, *images]
            if history_text:
                user_content[0]["text"] = history_text + user_content[0]["text"]
        else:
            user_content = history_text + prompt_text

        agent = get_agent()
        async with chat_reasoning():
            result = await agent.run(user_content, deps=deps)
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