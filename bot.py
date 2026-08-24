import re
import asyncio
import logging
import os
from pathlib import Path
import discord
from discord.ext import commands
from config import TOKEN, COMMAND_PREFIX
from core.db import init_db, track_user, get_monitored_channels
from core.memory import warmup_cognee
from services.monitoring import monitoring_cycle, emergency_monitoring
from core.conversation import answer_question
from config import SKILLS_DIRS
from services.skills import SkillManager
import services.skills as skills_module

log = logging.getLogger("rag-bot")

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix=COMMAND_PREFIX, intents=intents, help_command=None)

@bot.event
async def on_ready():
    await init_db()
    log.info("Logged in as %s", bot.user)

    # Initialize Skill Manager
    skills_module.skill_manager = SkillManager(SKILLS_DIRS)
    log.info(f"Loaded {len(skills_module.skill_manager.skills)} Agent Skills.")

    cogs_dir = Path(__file__).parent / "cogs"
    for filename in os.listdir(cogs_dir):
        if filename.endswith(".py") and not filename.startswith("__"):
            try:
                await bot.load_extension(f"cogs.{filename[:-3]}")
                log.info(f"Loaded cog: {filename}")
            except Exception as e:
                log.exception(f"Failed to load cog {filename}: {e}")

    asyncio.create_task(warmup_cognee())

    if not monitoring_cycle.is_running():
        monitoring_cycle.start(bot)
        log.info("Monitoring cycle started.")

@bot.event
async def on_message(message: discord.Message):
    if message.author.bot or message.webhook_id: return
    await track_user(message)
    await bot.process_commands(message)

    if message.guild is not None:
        try:
            monitored = await get_monitored_channels()
            for mc in monitored:
                if mc["channel_id"] != message.channel.id: continue
                if not mc["keywords"]: continue

                keywords = [kw.strip().lower() for kw in mc["keywords"].split(",")]
                message_lower = message.content.lower()
                for keyword in keywords:
                    if keyword and re.search(rf'\b{re.escape(keyword)}\b', message_lower):
                        log.info("Keyword '%s' detected in channel %s. Triggering emergency monitoring.", keyword, message.channel.name)
                        asyncio.create_task(emergency_monitoring(bot, message.channel, mc, message))
                        break
                break
        except Exception as e:
            log.exception("Keyword check failed: %s", e)

    is_mention = bot.user in message.mentions
    is_reply_to_bot = False
    if message.reference and message.reference.message_id:
        if message.reference.resolved and message.reference.resolved.author == bot.user:
            is_reply_to_bot = True

    if message.guild is None:
        await answer_question(bot, message, message.content)
        return

    if not (is_mention or is_reply_to_bot): return

    question = message.content
    if is_mention:
        question = re.sub(rf"<@!?{bot.user.id}>\s*[,.:;]?\s*", "", question).strip()

    if not question and not message.attachments:
        await message.reply("How can I help you?")
        return

    await answer_question(bot, message, question)

@bot.event
async def on_command_error(ctx: commands.Context, error: Exception):
    if isinstance(error, commands.MissingRequiredArgument):
        await ctx.reply("Missing required argument.")
        return
    if isinstance(error, commands.CheckFailure):
        await ctx.reply("You need `Manage Messages` permission in this server to use that command.")
        return
    if isinstance(error, commands.CommandNotFound): return
    log.error("Unhandled command error", exc_info=error)
    try:
        await ctx.reply("Unexpected error. Check the console/logs.")
    except Exception:
        pass