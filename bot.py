import re
import asyncio
import logging
import os
from pathlib import Path
import discord
from discord.ext import commands
from config import TOKEN, COMMAND_PREFIX
from core.db import init_db, track_user, get_monitored_channels, is_channel_permitted
from core.memory import warmup_memory
from services.monitoring import monitoring_scheduler, emergency_monitoring
from core.conversation import answer_question
from config import SKILLS_DIRS
from services.skills import SkillManager
import services.skills as skills_module
from services.game_db import load_game_database

log = logging.getLogger("rag-bot")

intents = discord.Intents.default()
intents.message_content = True

class MakerDroneBot(commands.Bot):
    async def setup_hook(self) -> None:
        await init_db()

        skills_module.skill_manager = SkillManager(SKILLS_DIRS)
        log.info("Loaded %s Agent Skills.", len(skills_module.skill_manager.skills))

        cogs_dir = Path(__file__).parent / "cogs"
        for filename in os.listdir(cogs_dir):
            if filename.endswith(".py") and not filename.startswith("__"):
                try:
                    await self.load_extension(f"cogs.{filename[:-3]}")
                    log.info("Loaded cog: %s", filename)
                except Exception:
                    log.exception("Failed to load cog %s", filename)

        await asyncio.to_thread(load_game_database)
        asyncio.create_task(warmup_memory())


bot = MakerDroneBot(command_prefix=COMMAND_PREFIX, intents=intents, help_command=None)

@bot.event
async def on_ready() -> None:
    log.info("Logged in as %s", bot.user)
    if not monitoring_scheduler.is_running():
        monitoring_scheduler.start(bot)
        log.info("Monitoring scheduler started.")

async def _evaluate_keyword_triggers(bot_instance: commands.Bot, message: discord.Message) -> None:
    if message.guild is None:
        return
    try:
        monitored = await get_monitored_channels()
        for mc in monitored:
            if mc["channel_id"] != message.channel.id:
                continue
            if not mc["keywords"]:
                continue
            keywords = [kw.strip().lower() for kw in mc["keywords"].split(",")]
            message_lower = message.content.lower()
            for keyword in keywords:
                if keyword and re.search(rf'\b{re.escape(keyword)}\b', message_lower):
                    log.info("Keyword '%s' detected in channel %s. Triggering emergency monitoring.", keyword, message.channel.name)
                    asyncio.create_task(emergency_monitoring(bot_instance, message.channel, mc, message))
                    break
            break
    except Exception as e:
        log.exception("Keyword check failed: %s", e)

def _is_reply_to_bot(message: discord.Message, bot_user: discord.ClientUser) -> bool:
    if message.reference and message.reference.message_id:
        if message.reference.resolved and message.reference.resolved.author == bot_user:
            return True
    return False

async def _route_chat_request(bot_instance: commands.Bot, message: discord.Message, can_speak: bool) -> None:
    is_mention = bot_instance.user in message.mentions
    is_reply = _is_reply_to_bot(message, bot_instance.user)

    if message.guild is None:
        if not can_speak:
            log.info("DM from %s dropped: channel not in talk whitelist.", message.author.display_name)
            return
        await answer_question(bot_instance, message, message.content)
        return

    if not (is_mention or is_reply):
        return

    if not can_speak:
        log.info("Chat request dropped: #%s is not a permitted talk channel.", message.channel.name)
        return

    question = message.content
    if is_mention:
        question = re.sub(rf"<@!?{bot_instance.user.id}>\s*[,.:;]?\s*", "", question).strip()

    if not question and not message.attachments:
        await message.reply("How can I help you?")
        return

    await answer_question(bot_instance, message, question)

@bot.event
async def on_message(message: discord.Message) -> None:
    if message.author.bot or message.webhook_id:
        return

    await track_user(message)

    can_speak = await is_channel_permitted(message.channel.id)
    if can_speak:
        await bot.process_commands(message)

    await _evaluate_keyword_triggers(bot, message)
    await _route_chat_request(bot, message, can_speak)

@bot.event
async def on_command_error(ctx: commands.Context, error: Exception) -> None:
    if isinstance(error, commands.CommandNotFound):
        return
    if isinstance(error, commands.MissingRequiredArgument):
        await ctx.reply("Missing required argument.")
        return
    if isinstance(error, commands.UserInputError):
        usage_hint = f"\nUsage: `{COMMAND_PREFIX}{ctx.command.usage}`" if ctx.command and ctx.command.usage else ""
        await ctx.reply(f"{error}{usage_hint}")
        return
    if isinstance(error, commands.CheckFailure):
        await ctx.reply("You need `Manage Messages` permission in this server to use that command.")
        return
    log.error("Unhandled command error", exc_info=error)
    try:
        await ctx.reply("Unexpected error. Check the console/logs.")
    except Exception:
        pass