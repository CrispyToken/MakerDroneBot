import discord
from discord.ext import commands
import aiohttp
from config import COMMAND_PREFIX, LLM_BASE_URL, LLM_API_KEY, LLM_MODEL_ID, DISCORD_CHAR_LIMIT


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

    @commands.command(name="status", help="Shows LLM backend and Cognee status.")
    async def status_cmd(self, ctx: commands.Context):
        llm_ok = True;
        llm_error = None
        try:
            timeout = aiohttp.ClientTimeout(total=10)
            headers = {"Authorization": f"Bearer {LLM_API_KEY}"} if LLM_API_KEY else {}
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(f"{LLM_BASE_URL}/models", headers=headers) as response:
                    llm_ok = response.status == 200
                    if not llm_ok: llm_error = await response.text()
        except Exception as e:
            llm_ok = False;
            llm_error = str(e)

        status = (
            f"LLM Base URL: `{LLM_BASE_URL}`\n"
            f"LLM reachable: `{'yes' if llm_ok else 'no'}`\n"
            f"Model ID: `{LLM_MODEL_ID}`\n"
            f"Discord char limit: `{DISCORD_CHAR_LIMIT}`\n"
            f"Cognee Datasets: `event_horizon`, `event_horizon_dynamic`"
        )
        if not llm_ok and llm_error: status += f"\n`{llm_error[:300]}`"
        await ctx.reply(status)


async def setup(bot):
    await bot.add_cog(GeneralCog(bot))