import os
import re
import asyncio
import discord
from discord.ext import commands
import aiosqlite
import cognee
import logging
from datetime import datetime, timezone
from config import INGEST_DIR, INGEST_EXTENSIONS, DB_PATH
from core.memory import load_ingest_hashes, save_ingest_hashes, compute_file_hash, cognee_in_background, \
    release_cognee_lock
from services.extractors import extract_text_from_file
from utils.formatting import split_for_discord
from utils.checks import is_staff
from core.db import ensure_user_in_db, get_user_profile, get_monitored_channels, get_config, set_config

log = logging.getLogger("rag-bot")


class StaffCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command(name="ingest", help="Ingest supported files from the local ingest/ folder into Cognee.")
    @is_staff()
    async def ingest_folder(self, ctx: commands.Context):
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
        deleted_results = []
        for deleted_key in deleted_keys:
            dataset_name = f"event_horizon__{deleted_key.replace(os.sep, '__').replace('/', '__').replace(' ', '_').lower()}"
            try:
                await cognee.forget(dataset_name=dataset_name)
                deleted_results.append(f"- `{deleted_key}` → removed from Cognee")
                del hash_records[deleted_key]
            except Exception as e:
                log.exception("Failed to forget deleted file %s", deleted_key)
                deleted_results.append(f"- `{deleted_key}` → forget failed: `{e}`")

        if not files_to_process and not deleted_results:
            await ctx.reply(f"All {len(files)} file(s) unchanged since last ingest. Nothing to do.")
            return

        status_parts = []
        if files_to_process: status_parts.append(f"**{len(files_to_process)}** file(s) to process")
        if files_skipped:
            unchanged_count = sum(1 for _, reason in files_skipped if reason == "unchanged")
            if unchanged_count: status_parts.append(f"**{unchanged_count}** unchanged (skipped)")
        if deleted_results: status_parts.append(f"**{len(deleted_results)}** deleted")

        await ctx.reply(
            f"Starting ingestion: {', '.join(status_parts)}.\nThis may take a while. I will report back when finished.")

        results = []
        for path, rel_path, file_hash, old_hash in files_to_process:
            safe_name = str(rel_path.with_suffix("")).replace(os.sep, "__").replace("/", "__").replace(" ", "_").lower()
            dataset_name = f"event_horizon__{safe_name}"
            try:
                data = await asyncio.to_thread(path.read_bytes)
                text = await asyncio.to_thread(extract_text_from_file, path.name, data)
                if not text.strip():
                    results.append(f"- `{rel_path}` → no readable text found")
                    continue
                if old_hash is not None:
                    try:
                        await cognee.forget(dataset_name=dataset_name)
                    except Exception:
                        log.warning("Could not forget old data for %s before re-ingest.", dataset_name)
                await cognee.remember(text, dataset_name=dataset_name)
                hash_records[str(rel_path)] = file_hash
                results.append(f"- `{rel_path}` → stored as `{dataset_name}`")
                log.info("Ingested and processed: %s", path.name)
            except Exception as e:
                log.exception("Failed to process %s for Cognee", path)
                results.append(f"- `{rel_path}` failed: `{e}`")
            finally:
                await release_cognee_lock()

        await asyncio.to_thread(save_ingest_hashes, hash_records)

        all_results = []
        if results:
            all_results.append("**Processed:**")
            all_results.extend(results)
        if files_skipped:
            unchanged = [f"- `{rp}`" for rp, reason in files_skipped if reason == "unchanged"]
            if unchanged: all_results.append(f"\n**Skipped (unchanged):** {len(unchanged)} file(s)")
        if deleted_results:
            all_results.append("\n**Removed:**")
            all_results.extend(deleted_results)

        summary = "Ingest results:\n" + "\n".join(all_results)
        for part in split_for_discord(summary):
            await ctx.reply(part)

    @commands.command(name="improve", help="Run Cognee's improve step to enrich and refine the knowledge graph.")
    @is_staff()
    async def improve_graph(self, ctx: commands.Context):
        await ctx.reply(
            "Starting knowledge graph improvement in the background.\nThis may take a while. I will report back when finished.")
        try:
            await cognee_in_background(cognee.improve)
            await ctx.reply("Knowledge graph improvement complete.")
            log.info("Cognee improve completed successfully.")
        except Exception as e:
            log.exception("Cognee improve failed")
            await ctx.reply(f"Knowledge graph improvement failed:\n`{e}`")
        finally:
            await release_cognee_lock()

    @commands.command(name="monitor", help="Manage channel monitoring. Usage: !monitor add/remove/list/setchannel")
    @is_staff()
    async def monitor_cmd(self, ctx: commands.Context, action: str = "list", *, args: str = ""):
        action = action.lower()
        if action == "list":
            monitored = await get_monitored_channels()
            if not monitored:
                await ctx.reply("No channels are currently being monitored.")
                return
            lines = ["**Monitored Channels:**"]
            for mc in monitored:
                kw = f" | Keywords: `{mc['keywords']}`" if mc["keywords"] else ""
                lines.append(f"- <#{mc['channel_id']}> ({mc['channel_name']}): {mc['reason']}{kw}")
            staff_ch = await get_config("staff_channel_id")
            if staff_ch:
                lines.append(f"\n**Alerts go to:** <#{staff_ch}>")
            else:
                lines.append("\n**Alerts go to:** Not configured. Use `!monitor setchannel #channel`")
            for part in split_for_discord("\n".join(lines)):
                await ctx.reply(part)

        elif action == "add":
            channel_match = re.search(r"<#(\d+)>", args)
            if not channel_match:
                await ctx.reply("Usage: `!monitor add #channel \"reason\" [keyword1,keyword2]`")
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
                    "Please provide a reason in quotes. Usage: `!monitor add #channel \"reason\" [keywords]`")
                return
            reason = reason_match.group(1)

            keywords = ""
            after_reason = args[reason_match.end():].strip()
            if after_reason: keywords = after_reason.strip()

            async with aiosqlite.connect(DB_PATH) as db:
                await db.execute("""
                                 INSERT INTO monitored_channels (channel_id, channel_name, reason, keywords, last_message_id)
                                 VALUES (?, ?, ?, ?, 0) ON CONFLICT(channel_id) DO
                                 UPDATE SET
                                     channel_name = excluded.channel_name,
                                     reason = excluded.reason,
                                     keywords = excluded.keywords
                                 """, (channel_id, channel.name, reason, keywords))
                await db.commit()
            kw_msg = f" with keywords `{keywords}`" if keywords else ""
            await ctx.reply(f"Now monitoring <#{channel_id}> for: {reason}{kw_msg}")

        elif action == "remove":
            channel_match = re.search(r"<#(\d+)>", args)
            if not channel_match:
                await ctx.reply("Usage: `!monitor remove #channel`")
                return
            channel_id = int(channel_match.group(1))
            async with aiosqlite.connect(DB_PATH) as db:
                await db.execute("DELETE FROM monitored_channels WHERE channel_id = ?", (channel_id,))
                await db.commit()
            await ctx.reply(f"Stopped monitoring <#{channel_id}>.")

        elif action == "setchannel":
            channel_match = re.search(r"<#(\d+)>", args)
            if not channel_match:
                await ctx.reply("Usage: `!monitor setchannel #channel`")
                return
            channel_id = channel_match.group(1)
            await set_config("staff_channel_id", channel_id)
            await ctx.reply(f"Monitoring alerts will now be sent to <#{channel_id}>.")

        else:
            await ctx.reply(
                "Unknown action. Use: `!monitor add`, `!monitor remove`, `!monitor list`, or `!monitor setchannel`")

    @commands.command(name="profile", help="View the bot's memory of a user.")
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

    @commands.command(name="note", help="Add a staff note to a user's profile. Usage: !note @User <text>")
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

    @commands.command(name="clearnotes", help="Clear all staff notes for a user.")
    @is_staff()
    async def clear_notes(self, ctx: commands.Context, member: discord.Member):
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute("UPDATE user_profiles SET staff_notes = '' WHERE user_id = ?", (member.id,))
            await db.commit()
        await ctx.reply(f"Cleared all notes for {member.display_name}.")


async def setup(bot):
    await bot.add_cog(StaffCog(bot))