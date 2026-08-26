import asyncio
import aiohttp
import logging
import discord
from datetime import datetime, timedelta, timezone
from discord.ext import tasks
import aiosqlite
import cognee

from config import (
    MONITOR_PROMPT_FILE, SERVER_RULES_FILE, MONITOR_INTERVAL_MINUTES,
    MONITOR_MAX_MESSAGES_PER_CHANNEL, LLM_BASE_URL, LLM_API_KEY, LLM_MODEL_ID, REQUEST_TIMEOUT, DB_PATH
)
from core.db import get_monitored_channels, get_config, set_config
from core.memory import cognee_in_background, release_cognee_lock

log = logging.getLogger("rag-bot")

monitoring_in_progress = False

# Persistent schedule checkpoint keys (stored in bot_config)
KEY_NEXT_RUN = "monitor_next_run_at"
KEY_INTERVAL = "monitor_interval_minutes"

# How often the scheduler checks whether the checkpoint has passed
SCHEDULER_TICK_SECONDS = 30


# ---------------------------------------------------------------------------
# Persistent checkpoint scheduling
# ---------------------------------------------------------------------------

async def get_interval_minutes() -> int:
    """Effective monitoring interval: runtime override if set, else env default."""
    stored = await get_config(KEY_INTERVAL)
    if stored:
        try:
            value = int(stored)
            if value >= 1:
                return value
        except ValueError:
            log.warning("Stored monitoring interval %r is invalid; using env default.", stored)
    return MONITOR_INTERVAL_MINUTES


async def set_interval_minutes(minutes: int) -> None:
    await set_config(KEY_INTERVAL, str(minutes))


async def get_next_run_at() -> datetime | None:
    stored = await get_config(KEY_NEXT_RUN)
    if not stored:
        return None
    try:
        return datetime.fromtimestamp(float(stored), tz=timezone.utc)
    except (ValueError, OSError):
        log.warning("Stored monitoring checkpoint %r is invalid; it will be reset.", stored)
        return None


async def schedule_next_from_now() -> None:
    """Anchor the next checkpoint relative to the present moment."""
    interval = await get_interval_minutes()
    next_run = datetime.now(timezone.utc) + timedelta(minutes=interval)
    await set_config(KEY_NEXT_RUN, str(next_run.timestamp()))
    log.info("Monitoring checkpoint set: next scan at %s (interval: %dm).", next_run.isoformat(), interval)


# ---------------------------------------------------------------------------
# Prompts & message fetching
# ---------------------------------------------------------------------------

def load_monitoring_prompt() -> str:
    try:
        if MONITOR_PROMPT_FILE.exists():
            prompt = MONITOR_PROMPT_FILE.read_text(encoding="utf-8").strip()
        else:
            log.warning("Monitoring prompt file not found at %s.", MONITOR_PROMPT_FILE)
            return ""
        if SERVER_RULES_FILE.exists():
            rules = SERVER_RULES_FILE.read_text(encoding="utf-8").strip()
        else:
            log.warning("Server rules file not found at %s.", SERVER_RULES_FILE)
            rules = "(No server rules file found.)"
        prompt = prompt.replace("{{SERVER_RULES}}", rules)
        return prompt
    except Exception as e:
        log.exception("Failed to read monitoring prompt.")
        return ""


async def fetch_new_messages_for_channel(channel: discord.TextChannel, last_message_id: int) -> list[discord.Message]:
    messages = []
    try:
        after = discord.Object(id=last_message_id) if last_message_id > 0 else None
        async for msg in channel.history(limit=MONITOR_MAX_MESSAGES_PER_CHANNEL, after=after, oldest_first=True):
            if msg.author.bot: continue
            if not msg.clean_content: continue
            messages.append(msg)
    except Exception as e:
        log.exception("Failed to fetch history for channel %s", channel.id)
    return messages


async def get_latest_message_id(channel: discord.TextChannel) -> int | None:
    """ID of the most recent message in the channel, if any."""
    try:
        async for msg in channel.history(limit=1):
            return msg.id
    except Exception:
        log.exception("Failed to fetch latest message for channel %s", channel.id)
    return None


# ---------------------------------------------------------------------------
# Context assembly
# ---------------------------------------------------------------------------

def build_channel_section(channel: discord.TextChannel, mc: dict, messages: list[discord.Message],
                          index: int, total: int) -> str:
    """Assemble one channel's new messages into a clearly framed context block."""
    lines = [f"=== MONITORED CHANNEL {index} OF {total}: #{channel.name} ==="]
    lines.append(f"Channel topic: {channel.topic if channel.topic else '(No topic set)'}")
    lines.append(f"Monitoring reason: {mc['reason']}")
    if mc["keywords"]:
        lines.append(f"Keywords to watch: {mc['keywords']}")
    lines.append(f"{len(messages)} new message(s) since the last scan, oldest first:")
    for msg in messages:
        timestamp = msg.created_at.strftime("%Y-%m-%d %H:%M UTC")
        lines.append(f"[{timestamp}] {msg.author.display_name}: {msg.clean_content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Evaluation (LLM verdict + action)
# ---------------------------------------------------------------------------

async def run_monitoring_evaluation(bot: discord.Client, channel_texts: list[str], is_emergency: bool = False):
    global monitoring_in_progress
    monitoring_prompt = load_monitoring_prompt()
    if not monitoring_prompt: return

    batched_context = "\n\n".join(channel_texts)
    full_prompt = f"{monitoring_prompt}\n\n=== MONITORED CHANNEL MESSAGES ===\n{batched_context}\n=== END MESSAGES ==="
    url = f"{LLM_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {LLM_API_KEY}"} if LLM_API_KEY else {}
    payload = {
        "model": LLM_MODEL_ID,
        "messages": [{"role": "user", "content": full_prompt}],
        "temperature": 0.1,
        "stream": False,
    }

    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    decision_text = ""
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, headers=headers, json=payload) as response:
            if response.status != 200:
                body = await response.text()
                log.error("Monitoring LLM call failed: %s %s", response.status, body[:300])
                return
            data = await response.json()
            decision_text = data["choices"][0]["message"]["content"].strip()

    log.info("Monitoring decision: %s", decision_text[:200])

    if decision_text.startswith("IGNORE"):
        log.info("Monitoring: Decision is IGNORE.")
    elif decision_text.startswith("REMEMBER:"):
        fact = decision_text[len("REMEMBER:"):].strip()
        if fact:
            try:
                await cognee_in_background(cognee.remember, fact, dataset_name="event_horizon_dynamic")
                log.info("Monitoring: Saved fact to dynamic memory.")
            except Exception as e:
                log.exception("Monitoring: Failed to save fact: %s", e)
    elif decision_text.startswith("ALERT:"):
        alert_message = decision_text[len("ALERT:"):].strip()
        staff_channel_id = await get_config("staff_channel_id")
        if staff_channel_id:
            staff_channel = bot.get_channel(int(staff_channel_id))
            if staff_channel:
                try:
                    prefix = "⚠️ **Monitoring Alert** (keyword trigger)\n" if is_emergency else "⚠️ **Monitoring Alert**\n"
                    await staff_channel.send(f"{prefix}{alert_message}")
                    log.info("Monitoring: Alert sent to staff channel.")
                except Exception as e:
                    log.exception("Monitoring: Failed to send alert: %s", e)
        else:
            log.warning("Monitoring: No staff channel configured.")


# ---------------------------------------------------------------------------
# Scheduled cycle
# ---------------------------------------------------------------------------

async def run_monitoring_cycle(bot: discord.Client) -> None:
    """One scheduled monitoring pass over all active monitored channels."""
    global monitoring_in_progress
    if monitoring_in_progress:
        log.info("Monitoring cycle: Skipping, another session is in progress.")
        return

    monitored = await get_monitored_channels()
    if not monitored:
        return

    monitoring_in_progress = True
    try:
        scan_results = []      # (channel, config, new_messages)
        baseline_updates = {}  # channel_id -> latest message id (first-scan baseline)

        for mc in monitored:
            channel = bot.get_channel(mc["channel_id"])
            if channel is None:
                try:
                    channel = await bot.fetch_channel(mc["channel_id"])
                except Exception:
                    log.warning("Monitoring: Could not access channel %s", mc["channel_id"])
                    continue

            new_messages = await fetch_new_messages_for_channel(channel, mc["last_message_id"])
            if not new_messages:
                # First scan of a channel with nothing eligible: set a baseline so we
                # don't re-pull the full history window on every subsequent cycle.
                if mc["last_message_id"] == 0:
                    latest_id = await get_latest_message_id(channel)
                    if latest_id:
                        baseline_updates[mc["channel_id"]] = latest_id
                        log.info("Monitoring: First scan of #%s found no eligible messages; baseline set to %s.",
                                 channel.name, latest_id)
                continue

            scan_results.append((channel, mc, new_messages))

        # LLM is only called when at least one channel has new messages.
        if scan_results:
            total = len(scan_results)
            sections = [
                build_channel_section(channel, mc, messages, index, total)
                for index, (channel, mc, messages) in enumerate(scan_results, start=1)
            ]
            await run_monitoring_evaluation(bot, sections, is_emergency=False)

        # Persist scan positions only after a successful pass.
        async with aiosqlite.connect(DB_PATH) as db:
            for channel, mc, messages in scan_results:
                await db.execute(
                    "UPDATE monitored_channels SET last_message_id = ? WHERE channel_id = ?",
                    (messages[-1].id, mc["channel_id"]),
                )
            for channel_id, latest_id in baseline_updates.items():
                await db.execute(
                    "UPDATE monitored_channels SET last_message_id = ? WHERE channel_id = ?",
                    (latest_id, channel_id),
                )
            await db.commit()
    except Exception as e:
        log.exception("Monitoring cycle failed: %s", e)
    finally:
        monitoring_in_progress = False
        await release_cognee_lock()
        log.info("Monitoring cycle: Lock released.")


@tasks.loop(seconds=SCHEDULER_TICK_SECONDS)
async def monitoring_scheduler(bot: discord.Client):
    """Checkpoint-based scheduler.

    The next-run timestamp lives in the database, so bot restarts never reset
    the schedule. If the stored checkpoint is already in the past when the bot
    starts (downtime), a catch-up scan runs immediately and the next checkpoint
    is re-anchored relative to the present moment.
    """
    try:
        next_run = await get_next_run_at()
        now = datetime.now(timezone.utc)

        if next_run is None:
            # First ever start: create the initial checkpoint, no immediate scan.
            await schedule_next_from_now()
            return

        if now >= next_run:
            if now - next_run > timedelta(minutes=1):
                log.info("Monitoring: checkpoint %s has passed; running catch-up scan.", next_run.isoformat())
            await run_monitoring_cycle(bot)
            await schedule_next_from_now()
    except Exception as e:
        log.exception("Monitoring scheduler tick failed: %s", e)


# ---------------------------------------------------------------------------
# Emergency (keyword) monitoring — unchanged behavior
# ---------------------------------------------------------------------------

async def emergency_monitoring(bot: discord.Client, channel: discord.TextChannel, channel_config: dict,
                                 trigger_message: discord.Message):
    global monitoring_in_progress
    if monitoring_in_progress:
        log.info("Emergency monitoring: Skipping, another session is in progress.")
        return

    monitoring_in_progress = True
    try:
        messages = []
        async for msg in channel.history(limit=MONITOR_MAX_MESSAGES_PER_CHANNEL, oldest_first=True):
            if msg.author.bot: continue
            if not msg.clean_content: continue
            messages.append(msg)
        if not messages: return

        section = f"#{channel.name}\n"
        topic_str = channel.topic if channel.topic else "(No topic set)"
        section += f"Channel topic: {topic_str}\n"
        section += f"Monitoring reason: {channel_config['reason']}\n"
        if channel_config["keywords"]:
            section += f"Keywords to watch: {channel_config['keywords']}\n"
        section += f"Trigger: A keyword was detected in a message by {trigger_message.author.display_name}.\n"
        section += "\nMessages:\n"
        for msg in messages:
            section += f"[{msg.author.display_name}]: {msg.clean_content}\n"

        await run_monitoring_evaluation(bot, [section], is_emergency=True)

        if messages:
            async with aiosqlite.connect(DB_PATH) as db:
                await db.execute("UPDATE monitored_channels SET last_message_id = ? WHERE channel_id = ?",
                                 (messages[-1].id, channel.id))
                await db.commit()
    except Exception as e:
        log.exception("Emergency monitoring failed: %s", e)
    finally:
        monitoring_in_progress = False
        await release_cognee_lock()
        log.info("Emergency monitoring: Lock released.")