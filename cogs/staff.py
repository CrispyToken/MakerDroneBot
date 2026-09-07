import os
import re
import asyncio
import discord
from discord.ext import commands
import aiosqlite
import logging
from datetime import datetime, timezone
from config import INGEST_DIR, INGEST_EXTENSIONS, DB_PATH, COMMAND_PREFIX, LLM_INGEST_MODEL_PATH
from core.memory import load_ingest_hashes, save_ingest_hashes, compute_file_hash, ingest_document, forget_document
from services.extractors import extract_text_from_file
from utils.formatting import split_for_discord
from utils.checks import is_staff
from core.db import ensure_user_in_db, get_user_profile, get_monitored_channels, get_config, set_config
from pathlib import Path
from core.locks import llm_lock
from services.monitoring import (
    set_interval_minutes, schedule_next_from_now, get_interval_minutes, get_next_run_at
)

log = logging.getLogger("rag-bot")


def _dataset_name(rel_key: str) -> str:
    """Stable LightRAG doc ID for an ingest/ file (must match across ingest + forget)."""
    safe = str(Path(rel_key).with_suffix("")).replace(os.sep, "__").replace("/", "__").replace(" ", "_").lower()
    return f"event_horizon__{safe}"

class StaffCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command(name="ingest",
                      help="Ingest supported files from the local `ingest/` folder into memory.",
                      usage="ingest")
    @is_staff()
    async def ingest_folder(self, ctx: commands.Context):
        # --- Phase 1: scan & hash (no LLM involved, no lock) --------------
        files = []
        for path in INGEST_DIR.rglob("*"):
            if not path.is_file(): continue
            rel_path = path.relative_to(INGEST_DIR)
            if any(part.startswith(".") for part in rel_path.parts): continue
            if path.suffix.lower() not in INGEST_EXTENSIONS: continue
            files.append(path)
        files.sort()
        if not files:
            supported = ", ".join(sorted(INGEST_EXTENSIONS))
            await ctx.reply(f"No supported files found in `{INGEST_DIR}`.\nSupported extensions: `{supported}`")
            return
        hash_records = await asyncio.to_thread(load_ingest_hashes)
        files_to_process = []
        files_skipped = []
        current_file_keys = set()
        for path in files:
            rel_path = path.relative_to(INGEST_DIR)
            file_key = str(rel_path)
            current_file_keys.add(file_key)
            try:
                file_hash = await asyncio.to_thread(compute_file_hash, path)
            except Exception:
                log.exception("Failed to hash %s", path)
                files_skipped.append((rel_path, "hash error"))
                continue
            stored_hash = hash_records.get(file_key)
            if stored_hash == file_hash:
                files_skipped.append((rel_path, "unchanged"))
                continue
            files_to_process.append((path, rel_path, file_hash, stored_hash))
        deleted_keys = set(hash_records.keys()) - current_file_keys
        if not files_to_process and not deleted_keys:
            await ctx.reply(f"All {len(files)} file(s) unchanged since last ingest. Nothing to do.")
            return
        engine_manager = getattr(self.bot, "engine_manager", None)
        use_ingest_model = bool(engine_manager and LLM_INGEST_MODEL_PATH)
        status_parts = []
        if files_to_process: status_parts.append(f"**{len(files_to_process)}** file(s) to process")
        if files_skipped:
            unchanged_count = sum(1 for _, reason in files_skipped if reason == "unchanged")
            if unchanged_count: status_parts.append(f"**{unchanged_count}** unchanged (skipped)")
        if deleted_keys: status_parts.append(f"**{len(deleted_keys)}** deleted")
        model_note = " Switching to the dedicated ingestion model for the graph pass." if use_ingest_model else ""
        await ctx.reply(
            f"Starting ingestion: {', '.join(status_parts)}.\n"
            f"Extracting document text first; chat and monitoring will be paused while the knowledge graph is updated.{model_note}\n"
            "This may take a while. I will report back when finished.")
        # --- Phase 2: text extraction (CPU-only, still no lock) -----------
        # Docling parsing can take minutes on large PDFs; keeping it outside
        # the lock avoids blacking out chat/monitoring before any LLM work.
        ok_results = []
        fail_results = []
        extracted = []  # (rel_path, file_hash, old_hash, dataset_name, text)
        for path, rel_path, file_hash, old_hash in files_to_process:
            dataset_name = _dataset_name(str(rel_path))
            try:
                data = await asyncio.to_thread(path.read_bytes)
                text = await asyncio.to_thread(extract_text_from_file, path.name, data)
            except Exception as e:
                log.exception("Failed to process %s for memory", path)
                fail_results.append(f"- `{rel_path}` failed: `{e}`")
                continue
            if not text.strip():
                fail_results.append(f"- `{rel_path}` → no readable text found")
                continue
            extracted.append((rel_path, file_hash, old_hash, dataset_name, text))
        if not extracted and not deleted_keys:
            # Nothing survived extraction and nothing was deleted: no LLM work.
            summary = "Ingest results:\n" + "\n".join(fail_results)
            for part in split_for_discord(summary):
                await ctx.reply(part)
            return
        # --- Phase 3: locked graph pass ------------------------------------
        if llm_lock.locked():
            await ctx.reply("Another task is still running. Waiting for it to finish before ingesting…")
        restore_failed = False
        await llm_lock.acquire()
        try:
            if use_ingest_model:
                try:
                    await engine_manager.load_ingest_model()
                except Exception:
                    log.exception("Failed to load ingest model; restoring default model.")
                    try:
                        await engine_manager.load_default_model()
                    except Exception:
                        log.exception("Failed to restore default model after ingest-model failure.")
                        await ctx.reply(
                            "Failed to load the ingestion model, and the default model also failed to reload. "
                            "Check the logs; a bot restart may be required.")
                        return
                    await ctx.reply("Failed to load the ingestion model. Ingestion aborted.")
                    return
            try:
                deleted_results = []
                for deleted_key in deleted_keys:
                    dataset_name = _dataset_name(deleted_key)
                    try:
                        await forget_document(dataset_name)
                        deleted_results.append(f"- `{deleted_key}` → removed from memory.")
                        del hash_records[deleted_key]
                    except Exception as e:
                        log.exception("Failed to forget deleted file %s", deleted_key)
                        deleted_results.append(f"- `{deleted_key}` → forget failed: `{e}`")
                for rel_path, file_hash, old_hash, dataset_name, text in extracted:
                    try:
                        if old_hash is not None:
                            try:
                                await forget_document(dataset_name)
                            except Exception:
                                log.warning("Could not forget old data for %s before re-ingest.", dataset_name)
                        await ingest_document(text, doc_id=dataset_name)
                        hash_records[str(rel_path)] = file_hash
                        ok_results.append(f"- `{rel_path}` → stored as `{dataset_name}`")
                        log.info("Ingested and processed: %s", rel_path)
                    except Exception as e:
                        log.exception("Failed to ingest %s", rel_path)
                        fail_results.append(f"- `{rel_path}` failed: `{e}`")
                await asyncio.to_thread(save_ingest_hashes, hash_records)
            finally:
                if use_ingest_model:
                    try:
                        await engine_manager.load_default_model()
                    except Exception:
                        log.exception("Failed to restore default model after ingestion.")
                        restore_failed = True
        finally:
            llm_lock.release()
        # --- Phase 4: report ------------------------------------------------
        for rel_path, reason in files_skipped:
            if reason != "unchanged":
                fail_results.append(f"- `{rel_path}` skipped ({reason})")
        all_results = []
        if ok_results:
            all_results.append("**Processed:**")
            all_results.extend(ok_results)
        if fail_results:
            all_results.append("\n**Failed:**")
            all_results.extend(fail_results)
        unchanged_count = sum(1 for _, reason in files_skipped if reason == "unchanged")
        if unchanged_count:
            all_results.append(f"\n**Skipped (unchanged):** {unchanged_count} file(s)")
        if deleted_results:
            all_results.append("\n**Removed:**")
            all_results.extend(deleted_results)
        summary = "Ingest results:\n" + "\n".join(all_results)
        for part in split_for_discord(summary):
            await ctx.reply(part)
        if use_ingest_model:
            if restore_failed:
                await ctx.reply(
                    "Ingestion finished, but the default model FAILED to reload. "
                    "Check the logs; a bot restart may be required.")
            else:
                await ctx.reply("Ingestion complete. Default model restored, chat and monitoring resumed.")
        else:
            await ctx.reply("Ingestion complete. Chat and monitoring resumed.")

    @commands.command(name="monitor",
                      help=(
                              f"Manage channel monitoring.\n"
                              f"`{COMMAND_PREFIX}monitor list` Show monitored channels, alert target, and schedule.\n"
                              f"`{COMMAND_PREFIX}monitor add #channel \"reason\" [keywords]` Start monitoring a channel.\n"
                              f"`{COMMAND_PREFIX}monitor remove #channel` Stop monitoring (scan position preserved).\n"
                              f"`{COMMAND_PREFIX}monitor setchannel #channel` Set where alerts are sent.\n"
                              f"`{COMMAND_PREFIX}monitor setinterval <minutes>` Set the scan interval (1–1440)."
                      ),
                      usage="monitor <add | remove | list | setchannel | setinterval>")
    @is_staff()
    async def monitor_cmd(self, ctx: commands.Context, action: str = "list", *, args: str = ""):
        action = action.lower()

        if action == "list":
            monitored = await get_monitored_channels()
            interval = await get_interval_minutes()
            next_run = await get_next_run_at()
            lines = []
            if not monitored:
                lines.append("No channels are currently being monitored.")
            else:
                lines.append("**Monitored Channels:**")
                for mc in monitored:
                    kw = f" | Keywords: `{mc['keywords']}`" if mc["keywords"] else ""
                    lines.append(f"- <#{mc['channel_id']}> ({mc['channel_name']}): {mc['reason']}{kw}")
            staff_ch = await get_config("staff_channel_id")
            if staff_ch:
                lines.append(f"\n**Alerts go to:** <#{staff_ch}>")
            else:
                lines.append(f"\n**Alerts go to:** Not configured. Use `{COMMAND_PREFIX}monitor setchannel #channel`")
            if next_run:
                lines.append(
                    f"**Schedule:** every {interval}m. Next scan at {next_run.strftime('%Y-%m-%d %H:%M UTC')}")
            else:
                lines.append(f"**Schedule:** every {interval}m. Checkpoint not yet created")
            for part in split_for_discord("\n".join(lines)):
                await ctx.reply(part)

        elif action == "add":
            channel_match = re.search(r"<#(\d+)>", args)
            if not channel_match:
                await ctx.reply(f"Usage: `{COMMAND_PREFIX}monitor add #channel \"reason\" [keyword1,keyword2]`")
                return
            channel_id = int(channel_match.group(1))
            channel = self.bot.get_channel(channel_id)
            if channel is None:
                try:
                    channel = await self.bot.fetch_channel(channel_id)
                except Exception:
                    await ctx.reply("Could not access that channel.")
                    return
            reason_match = re.search(r'"([^"]+)"', args)
            if not reason_match:
                await ctx.reply(
                    f"Please provide a reason in quotes. Usage: `{COMMAND_PREFIX}monitor add #channel \"reason\" [keywords]`")
                return
            reason = reason_match.group(1)
            keywords = ""
            after_reason = args[reason_match.end():].strip()
            if after_reason: keywords = after_reason.strip()

            async with aiosqlite.connect(DB_PATH) as db:
                # Detect re-adding a previously removed channel so we can resume scanning.
                db.row_factory = aiosqlite.Row
                async with db.execute(
                        "SELECT last_message_id FROM monitored_channels WHERE channel_id = ?",
                        (channel_id,),
                ) as cursor:
                    existing = await cursor.fetchone()
                await db.execute("""
                                 INSERT INTO monitored_channels (channel_id, channel_name, reason, keywords, last_message_id, active)
                                 VALUES (?, ?, ?, ?, 0, 1) ON CONFLICT(channel_id) DO
                                 UPDATE SET
                                     channel_name = excluded.channel_name,
                                     reason = excluded.reason,
                                     keywords = excluded.keywords,
                                     active = 1
                                 """, (channel_id, channel.name, reason, keywords))
                await db.commit()
            kw_msg = f" with keywords `{keywords}`" if keywords else ""
            if existing and existing["last_message_id"] > 0:
                await ctx.reply(f"Resuming monitoring of <#{channel_id}> where it left off, for: {reason}{kw_msg}")
            else:
                await ctx.reply(f"Now monitoring <#{channel_id}> for: {reason}{kw_msg}")

        elif action == "remove":
            channel_match = re.search(r"<#(\d+)>", args)
            if not channel_match:
                await ctx.reply(f"Usage: `{COMMAND_PREFIX}monitor remove #channel`")
                return
            channel_id = int(channel_match.group(1))
            async with aiosqlite.connect(DB_PATH) as db:
                await db.execute("UPDATE monitored_channels SET active = 0 WHERE channel_id = ?", (channel_id,))
                await db.commit()
            await ctx.reply(f"Stopped monitoring <#{channel_id}>. Its scan position is preserved for future re-adding.")

        elif action == "setchannel":
            channel_match = re.search(r"<#(\d+)>", args)
            if not channel_match:
                await ctx.reply(f"Usage: `{COMMAND_PREFIX}monitor setchannel #channel`")
                return
            channel_id = channel_match.group(1)
            await set_config("staff_channel_id", channel_id)
            await ctx.reply(f"Monitoring alerts will now be sent to <#{channel_id}>.")

        elif action == "setinterval":
            try:
                minutes = int(args.strip())
            except ValueError:
                await ctx.reply(f"Usage: `{COMMAND_PREFIX}monitor setinterval <minutes>`")
                return
            if not (1 <= minutes <= 1440):
                await ctx.reply("Interval must be between 1 and 1440 minutes.")
                return
            await set_interval_minutes(minutes)
            await schedule_next_from_now()
            await ctx.reply(f"Monitoring interval set to **{minutes}** minutes. Next scan checkpoint has been reset.")

        else:
            await ctx.reply(
                f"Unknown action. Use: `{COMMAND_PREFIX}monitor add`, `{COMMAND_PREFIX}monitor remove`, `{COMMAND_PREFIX}monitor list`, `{COMMAND_PREFIX}monitor setchannel`, or `{COMMAND_PREFIX}monitor setinterval`")

    @commands.command(name="profile", help="View the bot's memory of a user.", usage="profile [@User]")
    @is_staff()
    async def view_profile(self, ctx: commands.Context, member: discord.Member = None):
        member = member or ctx.author
        profile = await get_user_profile(member.id)
        if not profile:
            await ctx.reply(f"I have no memory of {member.display_name} yet.")
            return
        embed = discord.Embed(title=f"Profile: {profile['display_name']}", color=discord.Color.blue())
        embed.add_field(name="Username", value=f"@{profile['username']}", inline=True)
        embed.add_field(name="Roles", value=profile['roles'] or "None", inline=False)
        embed.add_field(name="Joined Server", value=profile['join_date'][:10] if profile['join_date'] else "Unknown",
                        inline=True)
        embed.add_field(name="Messages Seen", value=str(profile['message_count']), inline=True)
        embed.add_field(name="Staff Notes", value=profile['staff_notes'] or "None", inline=False)
        await ctx.reply(embed=embed)

    @commands.command(name="note", help="Add a staff note to a user's profile.", usage="note @User <text>")
    @is_staff()
    async def add_note(self, ctx: commands.Context, member: discord.Member, *, note: str):
        await ensure_user_in_db(member)
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        formatted_note = f"[{timestamp} by {ctx.author.display_name}] {note}"
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("""
                             UPDATE user_profiles
                             SET staff_notes = CASE WHEN staff_notes = '' THEN ? ELSE staff_notes || '\n' || ? END
                             WHERE user_id = ?
                             """, (formatted_note, formatted_note, member.id))
            await db.commit()
        await ctx.reply(f"Note added to {member.display_name}'s profile.")

    @commands.command(name="clearnotes", help="Clear all staff notes for a user.", usage="clearnotes @User")
    @is_staff()
    async def clear_notes(self, ctx: commands.Context, member: discord.Member):
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("UPDATE user_profiles SET staff_notes = '' WHERE user_id = ?", (member.id,))
            await db.commit()
        await ctx.reply(f"Cleared all notes for {member.display_name}.")


async def setup(bot):
    await bot.add_cog(StaffCog(bot))