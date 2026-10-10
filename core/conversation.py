import asyncio
import logging
import discord
from datetime import datetime
from core.agent import get_agent, BotDependencies
from core.db import get_user_profile, get_staff_role_ids
from utils.context import build_server_channel_list, build_conversation_turns, format_turn_line, resolve_replied_message
from utils.attachments import collect_image_attachments, collect_text_attachments
from utils.formatting import send_final_answer, sanitize_bot_style
from core.llm_reasoning import chat_reasoning
from core.console import print_user_line
from core.locks import llm_lock, track_llm_task, active_llm_task_label
from pydantic_ai import ImageUrl, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.exceptions import UnexpectedModelBehavior

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
        channel_id=message.channel.id,
        user_profile=profile,
        current_time_str=_format_current_time(),
        channel_name=channel_name,
        channel_topic=channel_topic,
        server_channels=server_channels,
        user_roles=user_roles,
        is_staff=is_staff,
    )

async def _assemble_user_content(message: discord.Message, bot_user_id: int,
                                 text_blocks: list[str], images: list[dict],
                                 replied_to: discord.Message | None = None,
                                 replied_text_blocks: list[str] | None = None,
                                 replied_images: list[dict] | None = None) -> str | list:
    parts: list[str | ImageUrl] = [format_turn_line(message, bot_user_id)]
    parts.extend(ImageUrl(url=img["image_url"]["url"]) for img in images)
    parts.extend(text_blocks)
    if replied_to is not None:
        author = replied_to.author.display_name
        parts.extend(f"[From the replied-to message by {author}]\n{block}"
                     for block in (replied_text_blocks or []))
        if replied_images:
            parts.append(f"[{len(replied_images)} image(s) attached by {author} in the replied-to message:]")
            parts.extend(ImageUrl(url=img["image_url"]["url"]) for img in replied_images)
    if len(parts) == 1:
        return parts[0]
    return parts

def _to_model_messages(turns: list[tuple[str, str | list]]) -> list[ModelRequest | ModelResponse] | None:
    if not turns:
        return None
    messages: list[ModelRequest | ModelResponse] = []
    for role, content in turns:
        if role == "assistant":
            messages.append(ModelResponse(parts=[TextPart(content=content)]))
        else:
            messages.append(ModelRequest(parts=[UserPromptPart(content=content)]))
    return messages

async def _execute_agent(message: discord.Message, user_content: str | list, deps: BotDependencies,
                         history_turns: list[tuple[str, str | list]]) -> str | None:
    agent = await get_agent()
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
                async with agent.iter(user_content, message_history=_to_model_messages(history_turns), deps=deps) as agent_run:
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
    except UnexpectedModelBehavior as e:
        if "exceeded before any response was generated" in str(e):
            log.warning("Model exhausted output budget on thinking: %s", e)
            return "I ran out of output space while thinking through that task. Please try breaking the request down into smaller steps."
        raise
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

        if not question.strip() and not images and not text_blocks and not replied_images and not replied_text_blocks:
            await message.reply("Provide a question or attach an image/text file.")
            return

        deps = await _gather_dependencies(bot, message)
        history_turns = await build_conversation_turns(message, bot.user.id)
        user_content = await _assemble_user_content(
            message, bot.user.id, text_blocks, images,
            replied_to, replied_text_blocks, replied_images
        )
        output = await _execute_agent(message, user_content, deps, history_turns)

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