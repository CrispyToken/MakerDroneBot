import discord
from discord.ext import commands
import aiohttp
from config import COMMAND_PREFIX, LLM_BASE_URL, LLM_API_KEY, LLM_MODEL_ID, DISCORD_CHAR_LIMIT


class GeneralCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command(name="help", help="Shows all available commands, or details for one command.",
                      usage="help [command]")
    async def help_cmd(self, ctx: commands.Context, *, command_name: str = ""):
        command_name = command_name.strip().lower()
        if command_name:
            command = self.bot.get_command(command_name)
            if command is None or command.hidden:
                await ctx.reply(f"Unknown command: `{COMMAND_PREFIX}{command_name}`")
                return
            sig = f" {command.signature}" if command.signature else ""
            syntax = f"{COMMAND_PREFIX}{command.name}{sig}"
            privilege = "Staff" if command.checks else "Everyone"
            full_help = (command.help or command.brief or "No description.").strip()
            embed = discord.Embed(title=f"{COMMAND_PREFIX}{command.name}", color=discord.Color.blue())
            embed.add_field(name="Syntax", value=f"`{syntax}`", inline=False)
            embed.add_field(name="Description", value=full_help, inline=False)
            embed.add_field(name="Access", value=privilege, inline=True)
            await ctx.reply(embed=embed)
            return

        embed = discord.Embed(
            title="Commands",
            description=f"Prefix: `{COMMAND_PREFIX}` · Run `{COMMAND_PREFIX}help <command>` for details.",
            color=discord.Color.blue(),
        )

        def format_command(command: commands.Command) -> str:
            desc = (command.help or command.brief or "No description.").strip().split("\n")[0]
            return f"`{COMMAND_PREFIX}{command.name}` : {desc}"

        general_lines = []
        staff_lines = []
        for command in sorted(self.bot.commands, key=lambda c: c.name):
            if command.hidden or command.name == "help":
                continue
            entry = format_command(command)
            if command.checks:
                staff_lines.append(entry)
            else:
                general_lines.append(entry)

        if general_lines:
            embed.add_field(name="General", value="\n".join(general_lines), inline=False)
        if staff_lines:
            embed.add_field(name="Staff", value="\n".join(staff_lines), inline=False)
        if not general_lines and not staff_lines:
            embed.add_field(name="Commands", value="No commands available.", inline=False)
        await ctx.reply(embed=embed)

    @commands.command(name="ping", help="Checks if the bot is alive.", usage="ping")
    async def ping(self, ctx: commands.Context):
        await ctx.reply(f"Pong. Gateway latency: {round(self.bot.latency * 1000)} ms")

    @commands.command(name="status", help="Shows LLM backend and memory status.", usage="status")
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
        )
        engine = getattr(self.bot, "engine_manager", None)
        if engine is not None:
            status += f"Loaded model: `{engine.current_model_rel_path}`\n"
        status += (
            f"Discord char limit: `{DISCORD_CHAR_LIMIT}`\n"
            f"Memory backend: `LightRAG` (workspaces: `knowledge`, `dynamic`)"
        )
        if not llm_ok and llm_error: status += f"\n`{llm_error[:300]}`"
        await ctx.reply(status)


async def setup(bot):
    await bot.add_cog(GeneralCog(bot))