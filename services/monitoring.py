import asyncio
import aiohttp
import logging
import discord
from discord.ext import tasks
import aiosqlite
import cognee
from config import (
    MONITOR_PROMPT_FILE, SERVER_RULES_FILE, MONITOR_INTERVAL_MINUTES,
    MONITOR_MAX_MESSAGES_PER_CHANNEL, LMSTUDIO_BASE_URL, REQUEST_TIMEOUT, DB_PATH
)
from core.db import get_monitored_channels, get_config
from core.memory import cognee_in_background, release_cognee_lock
from core.llm import ensure_model, get_auth_headers, MODEL_LOCK

log = logging.getLogger("rag-bot")
monitoring_in_progress = False


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


async def run_monitoring_evaluation(bot: discord.Client, channel_texts: list[str], is_emergency: bool = False):
    global monitoring_in_progress
    monitoring_prompt = load_monitoring_prompt()
    if not monitoring_prompt: return

    batched_context = "\n\n".join(channel_texts)
    full_prompt = f"{monitoring_prompt}\n\n=== MONITORED CHANNEL MESSAGES ===\n{batched_context}\n=== END MESSAGES ==="

    async with MODEL_LOCK:
        model_id = await ensure_model("fast")

    url = f"{LMSTUDIO_BASE_URL}/chat/completions"
    headers = get_auth_headers()
    payload = {
        "model": model_id,
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


@tasks.loop(minutes=MONITOR_INTERVAL_MINUTES)
async def monitoring_cycle(bot: discord.Client):
    global monitoring_in_progress
    if monitoring_in_progress:
        log.info("Monitoring cycle: Skipping, another session is in progress.")
        return

    monitored = await get_monitored_channels()
    if not monitored: return

    monitoring_in_progress = True
    try:
        all_channel_texts = []
        channel_message_map = {}
        for mc in monitored:
            channel = bot.get_channel(mc["channel_id"])
            if channel is None:
                try:
                    channel = await bot.fetch_channel(mc["channel_id"])
                except Exception:
                    log.warning("Monitoring: Could not access channel %s", mc["channel_id"])
                    continue
            new_messages = await fetch_new_messages_for_channel(channel, mc["last_message_id"])
            if not new_messages: continue

            channel_message_map[mc["channel_id"]] = new_messages
            section = f"#{channel.name}\n"
            topic_str = channel.topic if channel.topic else "(No topic set)"
            section += f"Channel topic: {topic_str}\n"
            section += f"Monitoring reason: {mc['reason']}\n"
            if mc["keywords"]:
                section += f"Keywords to watch: {mc['keywords']}\n"
            section += "\nMessages:\n"
            for msg in new_messages:
                section += f"[{msg.author.display_name}]: {msg.clean_content}\n"
            all_channel_texts.append(section)

        if all_channel_texts:
            await run_monitoring_evaluation(bot, all_channel_texts, is_emergency=False)

        async with aiosqlite.connect(DB_PATH) as db:
            for channel_id, messages in channel_message_map.items():
                if messages:
                    new_last_id = messages[-1].id
                    await db.execute("UPDATE monitored_channels SET last_message_id = ? WHERE channel_id = ?",
                                     (new_last_id, channel_id))
            await db.commit()
    except Exception as e:
        log.exception("Monitoring cycle failed: %s", e)
    finally:
        monitoring_in_progress = False
        await release_cognee_lock()
        log.info("Monitoring cycle: Lock released.")


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