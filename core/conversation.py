import asyncio
import logging
import discord
from datetime import datetime
from core.agent import get_agent, BotDependencies
from core.db import get_user_profile, get_staff_role_ids
from utils.context import build_server_channel_list, get_conversation_context, resolve_replied_message
from utils.attachments import collect_image_attachments, collect_text_attachments
from utils.formatting import send_final_answer, sanitize_bot_style
from core.llm_reasoning import chat_reasoning
from core.console import print_user_line
from core.locks import llm_lock, track_llm_task, active_llm_task_label
from config import CONVERSATION_MAX_HISTORY
from pydantic_ai import ModelResponse, TextPart, ToolCallPart

log = logging.getLogger("rag-bot")

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


async def _gather_dependencies(bot: discord.Client, message: discord.Message) -> BotDependencies:
    profile = await get_user_profile(message.author.id)
    channel_name = message.channel.name if message.guild else "Direct Message"
    channel_topic = getattr(message.channel, 'topic', None) or ""
    server_channels = build_server_channel_list(message.guild)
    user_roles: list[str] | None = None
    is_staff = False
    guild_id = message.guild.id if message.guild else 0

    if isinstance(message.author, discord.Member):
        user_roles = [role.name for role in message.author.roles if role.name != "@everyone"]
        staff_role_ids = await get_staff_role_ids()
        is_staff = bool({role.id for role in message.author.roles} & staff_role_ids)

    return BotDependencies(
        bot=bot,
        guild_id=guild_id,
        user_profile=profile,
        current_time_str=_format_current_time(),
        channel_name=channel_name,
        channel_topic=channel_topic,
        server_channels=server_channels,
        user_roles=user_roles,
        is_staff=is_staff,
    )

async def _build_prompt_text(message: discord.Message, bot_user_id: int, question: str, text_blocks: list[str]) -> str:
    prompt_text = question
    if text_blocks:
        prompt_text += "\n\n" + "\n\n".join(text_blocks)
    if message.author.id != bot_user_id:
        prompt_text = f"{message.author.display_name}: " + prompt_text
    return prompt_text

async def _assemble_user_content(message: discord.Message, bot_user_id: int, bot_display_name: str,
                                 question: str, text_blocks: list[str], images: list[dict],
                                 replied_to: discord.Message | None = None,
                                 replied_text_blocks: list[str] | None = None,
                                 replied_images: list[dict] | None = None) -> str | list:
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
    if replied_text_blocks and replied_to is not None:
        author = replied_to.author.display_name
        text_blocks = text_blocks + [f"[From the replied-to message by {author}]\n{block}"
                                     for block in replied_text_blocks]
    prompt_text = await _build_prompt_text(message, bot_user_id, question, text_blocks)
    replied_images = replied_images or []
    if images or replied_images:
        from pydantic_ai import ImageUrl
        content: list[str | ImageUrl] = [history_text + prompt_text]
        if images:
            content.append(
                f"[{len(images)} image(s) attached by {message.author.display_name} in the current message:]")
            content.extend(ImageUrl(url=img["image_url"]["url"]) for img in images)
        if replied_images:
            author = replied_to.author.display_name
            content.append(
                f"[{len(replied_images)} image(s) attached by {author} in the replied-to message:]")
            content.extend(ImageUrl(url=img["image_url"]["url"]) for img in replied_images)
        return content
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
                async with agent.iter(user_content, deps=deps) as agent_run:
                    async for node in agent_run:
                        model_response = getattr(node, 'model_response', None)
                        if isinstance(model_response, ModelResponse):
                            text_parts = [p.content for p in model_response.parts if isinstance(p, TextPart)]
                            has_tool_calls = any(isinstance(p, ToolCallPart) for p in model_response.parts)
                            reply_text = "\n".join(t for t in text_parts if t and t.strip()).strip()

                            if reply_text and has_tool_calls:
                                await message.reply(sanitize_bot_style(reply_text), mention_author=False)

                    if agent_run.result is not None:
                        return agent_run.result.output
                    return None
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
        replied_to = await resolve_replied_message(message)
        images, image_warnings = await collect_image_attachments(message)
        text_blocks, text_warnings = await collect_text_attachments(message)
        replied_images: list[dict] = []
        replied_text_blocks: list[str] = []
        if replied_to is not None and replied_to.attachments:
            replied_images, r_image_warnings = await collect_image_attachments(replied_to)
            replied_text_blocks, r_text_warnings = await collect_text_attachments(replied_to)
            for warning in r_image_warnings + r_text_warnings:
                log.info("Attachment warning (replied-to message): %s", warning)
        for warning in image_warnings + text_warnings:
            log.info("Attachment warning: %s", warning)

        user_question = question.strip()
        question_for_model = user_question
        if not question_for_model:
            if images or replied_images:
                question_for_model = "Describe the attached image(s) in detail."
            elif text_blocks or replied_text_blocks:
                question_for_model = "Review the attached text and answer any relevant request."
        if not question_for_model and not images and not text_blocks and not replied_images and not replied_text_blocks:
            await message.reply("Provide a question or attach an image/text file.")
            return

        deps = await _gather_dependencies(bot, message)
        user_content = await _assemble_user_content(
            message, bot.user.id, bot.user.display_name, question_for_model, text_blocks, images,
            replied_to, replied_text_blocks, replied_images
        )

        output = await _execute_agent(message, user_content, deps)
        if output is not None:
            await send_final_answer(message, output)

    except RuntimeError as e:
        log.exception("Model/request error")
        await message.reply(f"Model/request error:\n`{e}`")
    except Exception:
        log.exception("Answer handler failed")
        await message.reply("Sorry, something went wrong while processing that. Please try again in a moment.")
    finally:
        stop_typing.set()
        typing_task.cancel()
        try:
            await typing_task
        except asyncio.CancelledError:
            pass