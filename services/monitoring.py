import asyncio
import aiohttp
import logging
import discord
from datetime import datetime, timedelta, timezone
from pathlib import Path
from discord.ext import tasks
from core.locks import llm_lock, track_llm_task
from config import (
    MONITOR_PROMPT_FILE, SERVER_RULES_FILE, MONITOR_INTERVAL_MINUTES,
    MONITOR_MAX_MESSAGES_PER_CHANNEL, MONITOR_MAX_IMAGES_PER_CYCLE,
    ALLOWED_IMAGE_EXTENSIONS, LLM_BASE_URL, LLM_API_KEY, LLM_MODEL_ID, REQUEST_TIMEOUT
)
from core.db import get_monitored_channels, get_config, set_config, set_monitored_last_message_ids
from core.memory import memory_remember
from utils.attachments import collect_image_attachments, VISION_ENABLED
from core.console import print_completion

log = logging.getLogger("rag-bot")

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
    # Fetch a larger batch (3x) from the API to account for bot/empty messages
    # being filtered out, ensuring we get a full window of eligible messages.
    fetch_limit = MONITOR_MAX_MESSAGES_PER_CHANNEL * 3
    raw_messages = []
    try:
        # Fetch newest first to easily grab the latest context
        async for msg in channel.history(limit=fetch_limit, oldest_first=False):
            if msg.author.bot:
                continue
            if not msg.clean_content and not msg.attachments:
                continue
            raw_messages.append(msg)
            if len(raw_messages) >= MONITOR_MAX_MESSAGES_PER_CHANNEL:
                break
    except Exception:
        log.exception("Failed to fetch history for channel %s", channel.id)
        return []

    if not raw_messages:
        return []

    # raw_messages is newest-first. Reverse to chronological order.
    raw_messages.reverse()

    # Apply watermark logic.
    if last_message_id == 0:
        return raw_messages

    window_oldest_id = raw_messages[0].id

    if last_message_id < window_oldest_id:
        # Watermark is older than our window. Skip the ancient backlog, evaluate the whole window.
        return raw_messages
    else:
        # Watermark is within the window. Filter to strictly newer messages.
        return [msg for msg in raw_messages if msg.id > last_message_id]


async def get_latest_message_id(channel: discord.TextChannel) -> int | None:
    """ID of the most recent message in the channel, if any."""
    try:
        async for msg in channel.history(limit=1):
            return msg.id
    except Exception:
        log.exception("Failed to fetch latest message for channel %s", channel.id)
    return None

def _image_attachments(msg: discord.Message) -> list[discord.Attachment]:
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
                          index: int, total: int,
                          emergency_trigger_author: str | None = None) -> list[dict]:
    """Assemble one channel's messages into multimodal content parts.

    Each message contributes a text line (timestamp, author, content) followed
    immediately by that message's image parts. This adjacency is what lets the
    model attribute every image to its author and channel.

    emergency_trigger_author, when set, marks this section as the channel
    where a keyword trigger fired and names the triggering message's author.
    """
    header = []
    if emergency_trigger_author is not None:
        header.append(f"=== EMERGENCY MONITORING TRIGGERED IN #{channel.name} ===")
    header.extend([
        f"=== MONITORED CHANNEL {index} OF {total}: #{channel.name} ===",
        f"Channel topic: {channel.topic if channel.topic else '(No topic set)'}",
        f"Monitoring reason: {mc['reason']}",
    ])
    if mc["keywords"]:
        header.append(f"Keywords to watch: {mc['keywords']}")
    if emergency_trigger_author is not None:
        header.append(f"Trigger: A keyword was detected in a message by {emergency_trigger_author}.")
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

async def _scan_monitored_channels(bot: discord.Client, monitored: list[dict]) -> tuple[list, dict[int, int]]:
    """Shared unseen-message scan for scheduled and emergency runs.

    Pulls only messages newer than each channel's stored watermark and
    prepares them (text + image parts, shared cycle image budget).

    Returns (scan_results, baseline_updates):
      scan_results    : list of (channel, config, prepared_messages)
      baseline_updates: channel_id -> latest message id, for first scans
                        of channels with no eligible messages
    """
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

    return scan_results, baseline_updates


async def _persist_scan_positions(scan_results: list, baseline_updates: dict[int, int]) -> None:
    """Persist per-channel scan positions after a successful evaluation."""
    updates: dict[int, int] = {}
    for _, mc, prepared in scan_results:
        updates[mc["channel_id"]] = prepared[-1][0].id if prepared else mc["last_message_id"]
    for channel_id, latest_id in baseline_updates.items():
        updates[channel_id] = latest_id
    await set_monitored_last_message_ids(updates)

async def run_monitoring_evaluation(bot: discord.Client, sections: list[list[dict]], is_emergency: bool = False) -> None:
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
        "chat_template_kwargs": {"enable_thinking": True},
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
            message_data = data["choices"][0]["message"]
            decision_text = message_data["content"].strip()
        print_completion(message_data.get("reasoning_content") or "", decision_text, source="monitor")
        log.info("Monitoring decision: %s", decision_text[:200])

    if decision_text.startswith("IGNORE"):
        log.info("Monitoring: Decision is IGNORE.")
    elif decision_text.startswith("REMEMBER:"):
        fact = decision_text[len("REMEMBER:"):].strip()
        if fact:
            try:
                await memory_remember(fact)
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

async def run_monitoring_cycle(bot: discord.Client) -> bool:
    """One scheduled monitoring pass over all active monitored channels.

    Returns True if the cycle ran (or there was nothing to scan), False if
    it was skipped because the global LLM lock was held elsewhere (e.g. an
    ingestion is in progress). The scheduler uses this to decide whether to
    advance its checkpoint.
    """
    monitored = await get_monitored_channels()
    if not monitored:
        return True

    if llm_lock.locked():
        log.info("Monitoring cycle: skipping, LLM is busy with another task.")
        return False

    async with llm_lock:
        try:
            async with track_llm_task("monitoring"):
                scan_results, baseline_updates = await _scan_monitored_channels(bot, monitored)
                # LLM is only called when at least one channel has new messages.
                if scan_results:
                    total = len(scan_results)
                    sections = [
                        build_channel_section(channel, mc, prepared, index, total)
                        for index, (channel, mc, prepared) in enumerate(scan_results, start=1)
                    ]
                    await run_monitoring_evaluation(bot, sections, is_emergency=False)
                # Persist scan positions only after a successful pass.
                await _persist_scan_positions(scan_results, baseline_updates)
        except asyncio.CancelledError:
            log.warning("Monitoring cycle interrupted by staff command; scan position not advanced.")
            return False
        except Exception as e:
            log.exception("Monitoring cycle failed: %s", e)

    return True


@tasks.loop(seconds=SCHEDULER_TICK_SECONDS)
async def monitoring_scheduler(bot: discord.Client) -> None:
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
                log.info("Monitoring scheduler: catch-up scan (%s behind).", now - next_run)
            ran = await run_monitoring_cycle(bot)
            if ran:
                await schedule_next_from_now()
            else:
                log.info("Monitoring scheduler: cycle skipped; retrying on next tick.")
    except Exception as e:
        log.exception("Monitoring scheduler tick failed: %s", e)

async def emergency_monitoring(bot: discord.Client, channel: discord.TextChannel, channel_config: dict,
                               trigger_message: discord.Message) -> None:
    """Keyword-triggered immediate scan.

    Scans only the triggered channel, pulling only messages newer than its
    stored watermark, and persists the new scan position after the
    evaluation. This guarantees an already-evaluated message is never
    reported twice by later triggers.
    """
    if llm_lock.locked():
        log.info("Emergency monitoring: skipping, LLM is busy with another task.")
        return

    async with llm_lock:
        try:
            async with track_llm_task("emergency monitoring"):
                monitored = await get_monitored_channels()
                mc = next((m for m in monitored if m["channel_id"] == channel.id), None)
                if mc is None:
                    return

                scan_results, baseline_updates = await _scan_monitored_channels(bot, [mc])

                if not scan_results:
                    log.info("Emergency monitoring: no unseen eligible messages in #%s; nothing to evaluate.",
                             channel.name)
                    await _persist_scan_positions(scan_results, baseline_updates)
                    return

                sections = [
                    build_channel_section(
                        ch, cfg, prepared, 1, 1,
                        emergency_trigger_author=trigger_message.author.display_name,
                    )
                    for ch, cfg, prepared in scan_results
                ]
                await run_monitoring_evaluation(bot, sections, is_emergency=True)

                # Persist scan positions only after a successful pass.
                await _persist_scan_positions(scan_results, baseline_updates)
        except asyncio.CancelledError:
            log.warning("Emergency monitoring interrupted by staff command; scan position not advanced.")
            return
        except Exception as e:
            log.exception("Emergency monitoring failed: %s", e)