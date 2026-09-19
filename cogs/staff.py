import os
import re
import asyncio
import discord
from discord.ext import commands
import logging
from datetime import datetime, timezone
from config import INGEST_DIR, INGEST_EXTENSIONS, COMMAND_PREFIX, LLM_INGEST_MODEL_PATH
from core.memory import load_ingest_hashes, save_ingest_hashes, compute_file_hash, ingest_document, forget_document
from services.extractors import extract_text_from_file
from utils.formatting import split_for_discord
from utils.checks import is_staff
from core.db import (
    ensure_user_in_db, get_user_profile, get_monitored_channels, get_config, set_config,
    get_permitted_channels, add_permitted_channel, remove_permitted_channel,
    get_monitored_last_message_id, upsert_monitored_channel, deactivate_monitored_channel,
    append_staff_note, clear_staff_notes
)
from pathlib import Path
from core.locks import llm_lock, track_llm_task, interrupt_active_llm
from services.monitoring import (
    set_interval_minutes, schedule_next_from_now, get_interval_minutes, get_next_run_at,
    run_monitoring_cycle
)

log = logging.getLogger("rag-bot")

def _dataset_name(rel_key: str) -> str:
    safe = str(Path(rel_key).with_suffix("")).replace(os.sep, "_").replace("/", "_").replace(" ", "_").lower()
    return f"event_horizon__{safe}"

class StaffCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    # --- Ingest Helpers ------------------------------------------------------

    async def _scan_and_hash_files(self, ctx: commands.Context) -> tuple[list[tuple], list[tuple], set[str], dict[str, str], int]:
        files = []
        for path in INGEST_DIR.rglob("*"):
            if not path.is_file(): continue
            rel_path = path.relative_to(INGEST_DIR)
            if any(part.startswith(".") for part in rel_path.parts): continue
            if path.suffix.lower() not in INGEST_EXTENSIONS: continue
            files.append(path)
        files.sort()
        total_files = len(files)

        if not files:
            supported = ", ".join(sorted(INGEST_EXTENSIONS))
            await ctx.reply(f"No supported files found in `{INGEST_DIR}`.\nSupported extensions: `{supported}`")
            return [], [], set(), {}, 0

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
        return files_to_process, files_skipped, deleted_keys, hash_records, total_files

    async def _extract_texts(self, files_to_process: list[tuple]) -> tuple[list[tuple], list[str]]:
        extracted = []
        fail_results = []
        for path, rel_path, file_hash, old_hash in files_to_process:
            ds_name = _dataset_name(str(rel_path))
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
            extracted.append((rel_path, file_hash, old_hash, ds_name, text))
        return extracted, fail_results

    async def _execute_graph_ingestion(
        self, ctx: commands.Context, extracted: list[tuple], deleted_keys: set[str],
        hash_records: dict[str, str], use_ingest_model: bool
    ) -> tuple[list[str], list[str], list[str], bool, bool]:
        ok_results = []
        fail_results = []
        deleted_results = []
        interrupted = False
        restore_failed = False

        engine_manager = getattr(self.bot, "engine_manager", None)

        if llm_lock.locked():
            await ctx.reply("Another task is still running. Waiting for it to finish before ingesting…")

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
                        return ok_results, fail_results, deleted_results, True, True
                    await ctx.reply("Failed to load the ingestion model. Ingestion aborted.")
                    return ok_results, fail_results, deleted_results, True, False

            try:
                async with track_llm_task("ingest"):
                    for deleted_key in deleted_keys:
                        ds_name = _dataset_name(deleted_key)
                        try:
                            await forget_document(ds_name)
                            deleted_results.append(f"- `{deleted_key}` → removed from memory.")
                            del hash_records[deleted_key]
                            await asyncio.to_thread(save_ingest_hashes, hash_records)
                        except Exception as e:
                            log.exception("Failed to forget deleted file %s", deleted_key)
                            deleted_results.append(f"- `{deleted_key}` → forget failed: `{e}`")

                    for rel_path, file_hash, old_hash, ds_name, text in extracted:
                        try:
                            if old_hash is not None:
                                try:
                                    await forget_document(ds_name)
                                except Exception:
                                    log.warning("Could not forget old data for %s before re-ingest.", ds_name)
                            await ingest_document(text, doc_id=ds_name)
                            hash_records[str(rel_path)] = file_hash
                            await asyncio.to_thread(save_ingest_hashes, hash_records)
                            ok_results.append(f"- `{rel_path}` → stored as `{ds_name}`")
                            log.info("Ingested and processed: %s", rel_path)
                        except Exception as e:
                            log.exception("Failed to ingest %s", rel_path)
                            fail_results.append(f"- `{rel_path}` failed: `{e}`")
            except asyncio.CancelledError:
                log.warning("Ingestion interrupted by staff command. Restoring the default model.")
                interrupted = True
            finally:
                if use_ingest_model:
                    try:
                        await engine_manager.load_default_model()
                    except Exception:
                        log.exception("Failed to restore default model after ingestion.")
                        restore_failed = True
        finally:
            llm_lock.release()

        return ok_results, fail_results, deleted_results, interrupted, restore_failed

    def _format_ingest_report(
        self, files_skipped: list[tuple], ok_results: list[str], fail_results: list[str],
        deleted_results: list[str], interrupted: bool
    ) -> str:
        for rel_path, reason in files_skipped:
            if reason != "unchanged":
                fail_results.append(f"- `{rel_path}` skipped ({reason})")

        all_results = []
        if interrupted:
            all_results.append("**Interrupted:** the run was canceled by staff. Completed files are saved; "
                               "remaining files will be processed on the next ingest.")
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

        return "Ingest results:\n" + "\n".join(all_results)

    # --- Monitor & Talk Helpers ----------------------------------------------

    async def _handle_monitor_list(self, ctx: commands.Context) -> None:
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
            next_ts = int(next_run.timestamp())
            lines.append(f"**Schedule:** every {interval}m. Next scan at <t:{next_ts}:F> (<t:{next_ts}:R>)")
        else:
            lines.append(f"**Schedule:** every {interval}m. Checkpoint not yet created")
        for part in split_for_discord("\n".join(lines)):
            await ctx.reply(part)

    async def _handle_monitor_add(self, ctx: commands.Context, args: str) -> None:
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
            await ctx.reply(f"Please provide a reason in quotes. Usage: `{COMMAND_PREFIX}monitor add #channel \"reason\" [keywords]`")
            return
        reason = reason_match.group(1)
        keywords = ""
        after_reason = args[reason_match.end():].strip()
        if after_reason: keywords = after_reason.strip()

        previous_watermark = await get_monitored_last_message_id(channel_id)
        await upsert_monitored_channel(channel_id, channel.name, reason, keywords)

        kw_msg = f" with keywords `{keywords}`" if keywords else ""
        if previous_watermark and previous_watermark > 0:
            await ctx.reply(f"Resuming monitoring of <#{channel_id}> where it left off, for: {reason}{kw_msg}")
        else:
            await ctx.reply(f"Now monitoring <#{channel_id}> for: {reason}{kw_msg}")

    async def _handle_monitor_remove(self, ctx: commands.Context, args: str) -> None:
        channel_match = re.search(r"<#(\d+)>", args)
        if not channel_match:
            await ctx.reply(f"Usage: `{COMMAND_PREFIX}monitor remove #channel`")
            return
        channel_id = int(channel_match.group(1))
        await deactivate_monitored_channel(channel_id)
        await ctx.reply(f"Stopped monitoring <#{channel_id}>. Its scan position is preserved for future re-adding.")

    async def _handle_monitor_setchannel(self, ctx: commands.Context, args: str) -> None:
        channel_match = re.search(r"<#(\d+)>", args)
        if not channel_match:
            await ctx.reply(f"Usage: `{COMMAND_PREFIX}monitor setchannel #channel`")
            return
        channel_id = channel_match.group(1)
        await set_config("staff_channel_id", channel_id)
        await ctx.reply(f"Monitoring alerts will now be sent to <#{channel_id}>.")

    async def _handle_monitor_setinterval(self, ctx: commands.Context, args: str) -> None:
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

    async def _handle_monitor_now(self, ctx: commands.Context) -> None:
        if llm_lock.locked():
            await ctx.reply("I'm currently processing another task. Please try again in a minute.")
            return
        await ctx.reply("Starting a manual monitoring cycle…")
        ran = await run_monitoring_cycle(self.bot)
        if ran:
            await schedule_next_from_now()
            await ctx.reply("Manual monitoring cycle finished. The next scheduled scan checkpoint has been reset.")
        else:
            await ctx.reply("The monitoring cycle was skipped because the LLM became busy.")

    async def _handle_talk_list(self, ctx: commands.Context) -> None:
        permitted = await get_permitted_channels()
        if not permitted:
            await ctx.reply("No permitted talk channels configured. I will not speak anywhere, including DMs.")
            return
        lines = ["**Permitted talk channels:**"]
        for pc in permitted:
            lines.append(f"- <#{pc['channel_id']}> ({pc['channel_name']})")
        for part in split_for_discord("\n".join(lines)):
            await ctx.reply(part)

    async def _handle_talk_add(self, ctx: commands.Context, args: str) -> None:
        channel_match = re.search(r"<#(\d+)>", args)
        if not channel_match:
            await ctx.reply(f"Usage: `{COMMAND_PREFIX}talk add #channel`")
            return
        channel_id = int(channel_match.group(1))
        channel = self.bot.get_channel(channel_id)
        if channel is None:
            try:
                channel = await self.bot.fetch_channel(channel_id)
            except Exception:
                await ctx.reply("Could not access that channel.")
                return
        await add_permitted_channel(channel_id, channel.name)
        log.info("Talk permission granted: #%s (%s)", channel.name, channel_id)
        await ctx.reply(f"<#{channel_id}> is now a permitted talk channel.")

    async def _handle_talk_remove(self, ctx: commands.Context, args: str) -> None:
        channel_match = re.search(r"<#(\d+)>", args)
        if not channel_match:
            await ctx.reply(f"Usage: `{COMMAND_PREFIX}talk remove #channel`")
            return
        channel_id = int(channel_match.group(1))
        removed = await remove_permitted_channel(channel_id)
        log.info("Talk permission revoked: %s", channel_id)
        if removed:
            await ctx.reply(f"<#{channel_id}> is no longer a permitted talk channel.")
        else:
            await ctx.reply(f"<#{channel_id}> was not a permitted talk channel.")

    # --- Commands ------------------------------------------------------------

    @commands.command(name="ingest",
                      help="Ingest supported files from the local `ingest/` folder into memory.",
                      usage="ingest")
    @is_staff()
    async def ingest_folder(self, ctx: commands.Context) -> None:
        files_to_process, files_skipped, deleted_keys, hash_records, total_files = await self._scan_and_hash_files(ctx)
        if not files_to_process and not deleted_keys and not files_skipped:
            return

        if not files_to_process and not deleted_keys:
            await ctx.reply(f"All {total_files} file(s) unchanged since last ingest. Nothing to do.")
            return

        engine_manager = getattr(self.bot, "engine_manager", None)
        use_ingest_model = bool(engine_manager and LLM_INGEST_MODEL_PATH)

        status_parts = []
        if files_to_process: status_parts.append(f"**{len(files_to_process)}** file(s) to process")
        if files_skipped:
            unchanged_count = sum(1 for _, reason in files_skipped if reason == "unchanged")
            if unchanged_count: status_parts.append(f"**{unchanged_count}** unchanged (skipped)")
        if deleted_keys: status_parts.append(f"**{len(deleted_keys)}** deleted")

        await ctx.reply(
            f"Starting ingestion: {', '.join(status_parts)}.\n"
            "This may take a while. I will report back when finished.")

        extracted, fail_results = await self._extract_texts(files_to_process)

        if not extracted and not deleted_keys:
            summary = "Ingest results:\n" + "\n".join(fail_results)
            for part in split_for_discord(summary):
                await ctx.reply(part)
            return

        ok_results, extraction_fails, deleted_results, interrupted, restore_failed = await self._execute_graph_ingestion(
            ctx, extracted, deleted_keys, hash_records, use_ingest_model
        )
        fail_results.extend(extraction_fails)

        summary = self._format_ingest_report(files_skipped, ok_results, fail_results, deleted_results, interrupted)
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
                          f"`{COMMAND_PREFIX}monitor now` Run a monitoring cycle immediately.\n"
                          f"`{COMMAND_PREFIX}monitor list` Show monitored channels, alert target, and schedule.\n"
                          f"`{COMMAND_PREFIX}monitor add #channel \"reason\" [keywords]` Start monitoring a channel.\n"
                          f"`{COMMAND_PREFIX}monitor remove #channel` Stop monitoring (scan position preserved).\n"
                          f"`{COMMAND_PREFIX}monitor setchannel #channel` Set where alerts are sent.\n"
                          f"`{COMMAND_PREFIX}monitor setinterval <minutes>` Set the scan interval (1–1440)."
                      ),
                      usage="monitor <add | remove | list | setchannel | setinterval>")
    @is_staff()
    async def monitor_cmd(self, ctx: commands.Context, action: str = "list", *, args: str = "") -> None:
        action = action.lower()
        if action == "list":
            await self._handle_monitor_list(ctx)
        elif action == "add":
            await self._handle_monitor_add(ctx, args)
        elif action == "remove":
            await self._handle_monitor_remove(ctx, args)
        elif action == "setchannel":
            await self._handle_monitor_setchannel(ctx, args)
        elif action == "setinterval":
            await self._handle_monitor_setinterval(ctx, args)
        elif action == "now":
            await self._handle_monitor_now(ctx)
        else:
            await ctx.reply(
                f"Unknown action. Use: `{COMMAND_PREFIX}monitor add`, `{COMMAND_PREFIX}monitor remove`, `{COMMAND_PREFIX}monitor list`, `{COMMAND_PREFIX}monitor setchannel`, or `{COMMAND_PREFIX}monitor setinterval`")

    @commands.command(name="talk",
                      help=(
                          f"Control which channels the bot is allowed to speak in.\n"
                          f"`{COMMAND_PREFIX}talk add #channel` Grant speaking permission.\n"
                          f"`{COMMAND_PREFIX}talk remove #channel` Revoke speaking permission.\n"
                          f"`{COMMAND_PREFIX}talk list` Show all permitted channels."
                      ),
                      usage="talk <add | remove | list>")
    @is_staff()
    async def talk_cmd(self, ctx: commands.Context, action: str = "list", *, args: str = "") -> None:
        action = action.lower()
        if action == "list":
            await self._handle_talk_list(ctx)
        elif action == "add":
            await self._handle_talk_add(ctx, args)
        elif action == "remove":
            await self._handle_talk_remove(ctx, args)
        else:
            await ctx.reply(
                f"Unknown action. Use: `{COMMAND_PREFIX}talk add`, `{COMMAND_PREFIX}talk remove`, or `{COMMAND_PREFIX}talk list`")

    @commands.command(name="profile", help="View the bot's memory of a user.", usage="profile [@User]")
    @is_staff()
    async def view_profile(self, ctx: commands.Context, member: discord.Member | None = None) -> None:
        member = member or ctx.author
        profile = await get_user_profile(member.id)
        if not profile:
            await ctx.reply(f"I have no memory of {member.display_name} yet.")
            return
        embed = discord.Embed(title=f"Profile: {profile['display_name']}", color=discord.Color.blue())
        embed.add_field(name="Username", value=f"@{profile['username']}", inline=True)
        embed.add_field(name="Roles", value=profile['roles'] or "None", inline=False)
        join_val = "Unknown"
        if profile['join_date']:
            try:
                jd = datetime.fromisoformat(profile['join_date'])
                join_ts = int(jd.timestamp())
                join_val = f"<t:{join_ts}:d> (<t:{join_ts}:R>)"
            except Exception:
                join_val = profile['join_date'][:10]
        embed.add_field(name="Joined Server", value=join_val, inline=True)
        embed.add_field(name="Messages Seen", value=str(profile['message_count']), inline=True)
        embed.add_field(name="Staff Notes", value=profile['staff_notes'] or "None", inline=False)
        await ctx.reply(embed=embed)

    @commands.command(name="note", help="Add a staff note to a user's profile.", usage="note @User <text>")
    @is_staff()
    async def add_note(self, ctx: commands.Context, member: discord.Member, *, note: str) -> None:
        await ensure_user_in_db(member)
        ts = int(datetime.now(timezone.utc).timestamp())
        formatted_note = f"[<t:{ts}:d> by {ctx.author.display_name}] {note}"
        await append_staff_note(member.id, formatted_note)
        await ctx.reply(f"Note added to {member.display_name}'s profile.")

    @commands.command(name="clearnotes", help="Clear all staff notes for a user.", usage="clearnotes @User")
    @is_staff()
    async def clear_notes(self, ctx: commands.Context, member: discord.Member) -> None:
        await clear_staff_notes(member.id)
        await ctx.reply(f"Cleared all notes for {member.display_name}.")

    @commands.command(name="interrupt",
                      help="Cancel the currently running LLM task (chat, monitoring, or ingest). "
                           "The inference server and loaded model are unaffected.",
                      usage="interrupt")
    @is_staff()
    async def interrupt_cmd(self, ctx: commands.Context) -> None:
        label = interrupt_active_llm()
        if label is None:
            await ctx.reply("No active LLM task to interrupt.")
            return
        await ctx.reply(f"Interrupting the active LLM task ({label}).")

async def setup(bot):
    await bot.add_cog(StaffCog(bot))