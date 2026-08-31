import asyncio
import aiohttp
import logging
import discord
from datetime import datetime, timedelta, timezone
from pathlib import Path
from discord.ext import tasks
import aiosqlite
import cognee

from config import (
    MONITOR_PROMPT_FILE, SERVER_RULES_FILE, MONITOR_INTERVAL_MINUTES,
    MONITOR_MAX_MESSAGES_PER_CHANNEL, MONITOR_MAX_IMAGES_PER_CYCLE,
    ALLOWED_IMAGE_EXTENSIONS, LLM_BASE_URL, LLM_API_KEY, LLM_MODEL_ID, REQUEST_TIMEOUT, DB_PATH
)
from core.db import get_monitored_channels, get_config, set_config
from core.memory import cognee_in_background, release_cognee_lock
from utils.attachments import collect_image_attachments, VISION_ENABLED

log = logging.getLogger("rag-bot")

monitoring_in_progress = False

# Persistent schedule checkpoint keys (stored in bot_config)
KEY_NEXT_RUN = "monitor_next_run_at"
KEY_INTERVAL = "monitor_interval_minutes"

# How often the scheduler checks whether the checkpoint has passed
SCHEDULER_TICK_SECONDS = 30

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
            # Keep messages that have text OR attachments (images are evaluated separately).
            if not msg.clean_content and not msg.attachments: continue
            messages.append(msg)
    except Exception:
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

def _image_attachments(msg: discord.Message) -> list:
    """Image attachments on a message (mirrors the chat-side detection logic)."""
    return [
        att for att in msg.attachments
        if (att.content_type and att.content_type.startswith("image/"))
        or Path(att.filename or "").suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
    ]


async def collect_message_attachments(msg: discord.Message, budget: list[int]) -> tuple[list[dict], list[str]]:
    """Collect image content parts for one monitored message, honoring the shared cycle budget.

    budget is a one-element list [remaining_image_slots] shared across the whole scan.
    Returns (image_content_parts, marker_lines).
    """
    markers = []
    image_atts = _image_attachments(msg)
    if not image_atts:
        return [], markers

    names = ", ".join(att.filename or "image" for att in image_atts)

    if not VISION_ENABLED:
        markers.append(f"{msg.author.display_name} attached image(s), not analyzed (vision disabled): {names}")
        return [], markers

    if budget[0] <= 0:
        markers.append(f"{msg.author.display_name} attached image(s), not included (scan image cap reached): {names}")
        return [], markers

    blocks, _ = await collect_image_attachments(msg)
    if not blocks:
        markers.append(f"{msg.author.display_name} attached image(s) that could not be processed: {names}")
        return [], markers

    if len(blocks) > budget[0]:
        dropped = len(blocks) - budget[0]
        blocks = blocks[:budget[0]]
        markers.append(f"{dropped} additional image(s) from {msg.author.display_name} omitted (scan image cap).")
    budget[0] -= len(blocks)
    return blocks, markers


def build_channel_section(channel: discord.TextChannel, mc: dict, prepared_messages: list,
                          index: int, total: int) -> list[dict]:
    """Assemble one channel's messages into multimodal content parts.

    Each message contributes a text line (timestamp, author, content) followed
    immediately by that message's image parts — this adjacency is what lets the
    model attribute every image to its author and channel.
    """
    header = [
        f"=== MONITORED CHANNEL {index} OF {total}: #{channel.name} ===",
        f"Channel topic: {channel.topic if channel.topic else '(No topic set)'}",
        f"Monitoring reason: {mc['reason']}",
    ]
    if mc["keywords"]:
        header.append(f"Keywords to watch: {mc['keywords']}")
    header.append(f"{len(prepared_messages)} new message(s) since the last scan, oldest first:")

    parts = [{"type": "text", "text": "\n".join(header)}]
    for msg, image_blocks, markers in prepared_messages:
        timestamp = msg.created_at.strftime("%Y-%m-%d %H:%M UTC")
        text_content = msg.clean_content or "(no text content)"
        parts.append({"type": "text", "text": f"[{timestamp}] {msg.author.display_name}: {text_content}"})
        for marker in markers:
            parts.append({"type": "text", "text": f"[{timestamp}] {marker}"})
        parts.extend(image_blocks)
    return parts

async def run_monitoring_evaluation(bot: discord.Client, sections: list[list[dict]], is_emergency: bool = False):
    global monitoring_in_progress
    monitoring_prompt = load_monitoring_prompt()
    if not monitoring_prompt:
        return

    content_parts = [{"type": "text", "text": f"{monitoring_prompt}\n\n=== MONITORED CHANNEL MESSAGES ==="}]
    for section in sections:
        content_parts.extend(section)
    content_parts.append({"type": "text", "text": "=== END MESSAGES ==="})

    url = f"{LLM_BASE_URL}/chat/completions"
    headers = {"Authorization": f"Bearer {LLM_API_KEY}"} if LLM_API_KEY else {}
    payload = {
        "model": LLM_MODEL_ID,
        "messages": [{"role": "user", "content": content_parts}],
        "temperature": 0.1,
        "stream": False,
        # Monitoring verdicts are classification, not conversation — no thinking needed.
        "chat_template_kwargs": {"enable_thinking": False},
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
        image_budget = [MONITOR_MAX_IMAGES_PER_CYCLE]
        scan_results = []      # (channel, config, prepared_messages)
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

            prepared = []
            for msg in new_messages:
                blocks, markers = await collect_message_attachments(msg, image_budget)
                prepared.append((msg, blocks, markers))
            scan_results.append((channel, mc, prepared))

        # LLM is only called when at least one channel has new messages.
        if scan_results:
            total = len(scan_results)
            sections = [
                build_channel_section(channel, mc, prepared, index, total)
                for index, (channel, mc, prepared) in enumerate(scan_results, start=1)
            ]
            await run_monitoring_evaluation(bot, sections, is_emergency=False)

        # Persist scan positions only after a successful pass.
        async with aiosqlite.connect(DB_PATH) as db:
            for channel, mc, prepared in scan_results:
                await db.execute(
                    "UPDATE monitored_channels SET last_message_id = ? WHERE channel_id = ?",
                    (prepared[-1][0].id, mc["channel_id"]),
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
            if not msg.clean_content and not msg.attachments: continue
            messages.append(msg)
        if not messages:
            return

        image_budget = [MONITOR_MAX_IMAGES_PER_CYCLE]
        header = [
            f"=== EMERGENCY MONITORING: #{channel.name} ===",
            f"Channel topic: {channel.topic if channel.topic else '(No topic set)'}",
            f"Monitoring reason: {channel_config['reason']}",
        ]
        if channel_config["keywords"]:
            header.append(f"Keywords to watch: {channel_config['keywords']}")
        header.append(f"Trigger: A keyword was detected in a message by {trigger_message.author.display_name}.")
        header.append(f"{len(messages)} recent message(s), oldest first:")

        section = [{"type": "text", "text": "\n".join(header)}]
        for msg in messages:
            blocks, markers = await collect_message_attachments(msg, image_budget)
            timestamp = msg.created_at.strftime("%Y-%m-%d %H:%M UTC")
            text_content = msg.clean_content or "(no text content)"
            section.append({"type": "text", "text": f"[{timestamp}] {msg.author.display_name}: {text_content}"})
            for marker in markers:
                section.append({"type": "text", "text": f"[{timestamp}] {marker}"})
            section.extend(blocks)

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