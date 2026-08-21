import discord
from discord.ext import commands
import aiohttp
from config import COMMAND_PREFIX, LMSTUDIO_BASE_URL, FAST_MODEL_ID, DEEP_MODEL_ID, DISCORD_CHAR_LIMIT
from core.llm import get_loaded_model, get_auth_headers


class GeneralCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command(name="help", help="Shows all available commands.")
    async def help_cmd(self, ctx: commands.Context):
        embed = discord.Embed(
            title="MakerDrone Commands",
            description=f"All commands use the prefix `{COMMAND_PREFIX}`.",
            color=discord.Color.blue(),
        )
        general_lines = []
        staff_lines = []
        for command in sorted(self.bot.commands, key=lambda c: c.name):
            if command.hidden or command.name == "help": continue
            try:
                can_run = await command.can_run(ctx)
            except Exception:
                can_run = False
            if not can_run: continue

            desc = (command.help or command.brief or "No description.").strip()
            desc = desc.split("\n")[0]
            line = f"`{COMMAND_PREFIX}{command.name}` — {desc}"
            if command.checks:
                staff_lines.append(line)
            else:
                general_lines.append(line)

        if general_lines: embed.add_field(name="General", value="\n".join(general_lines), inline=False)
        if staff_lines: embed.add_field(name="Staff Only 🔒", value="\n".join(staff_lines), inline=False)
        if not general_lines and not staff_lines: embed.add_field(name="Commands", value="No commands available.",
                                                                  inline=False)
        await ctx.reply(embed=embed)

    @commands.command(name="ping", help="Checks if the bot is alive.")
    async def ping(self, ctx: commands.Context):
        await ctx.reply(f"Pong. Gateway latency: {round(self.bot.latency * 1000)} ms")

    @commands.command(name="status", help="Shows LM Studio and Cognee status.")
    async def status_cmd(self, ctx: commands.Context):
        lm_ok = True;
        lm_error = None
        currently_loaded = await get_loaded_model()
        try:
            timeout = aiohttp.ClientTimeout(total=10)
            headers = get_auth_headers()
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"{LMSTUDIO_BASE_URL}/models", headers=headers) as response:
                    lm_ok = response.status == 200
                    if not lm_ok: lm_error = await response.text()
        except Exception as e:
            lm_ok = False;
            lm_error = str(e)

        status = (
            f"LM Studio URL: `{LMSTUDIO_BASE_URL}`\n"
            f"LM Studio reachable: `{'yes' if lm_ok else 'no'}`\n"
            f"Fast model ID: `{FAST_MODEL_ID}`\n"
            f"Deep model ID: `{DEEP_MODEL_ID}`\n"
            f"Currently loaded: `{currently_loaded or 'None'}`\n"
            f"Discord char limit: `{DISCORD_CHAR_LIMIT}`\n"
            f"Cognee Datasets: `event_horizon`, `event_horizon_dynamic`"
        )
        if not lm_ok and lm_error: status += f"\n`{lm_error[:300]}`"
        await ctx.reply(status)


async def setup(bot):
    await bot.add_cog(GeneralCog(bot))