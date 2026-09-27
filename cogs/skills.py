import re
from discord.ext import commands
from config import SKILLS_DIRS
from utils.formatting import split_for_discord
from utils.checks import is_staff


class SkillsCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command(name="skills", help="List all installed Agent Skills.", usage="skills")
    @is_staff()
    async def list_skills(self, ctx: commands.Context):
        skills = []
        for d in SKILLS_DIRS:
            if not d.is_dir():
                continue
            for path in d.rglob("SKILL.md"):
                try:
                    text = path.read_text(encoding="utf-8")
                    match = re.match(r'^---\s*\n(.*?)\n---\s*\n(.*)$', text, re.DOTALL)
                    if not match:
                        continue
                    meta = match.group(1)
                    name = ""
                    description = ""
                    for line in meta.split('\n'):
                        if line.startswith('name:'):
                            name = line.split(':', 1)[1].strip().strip('"\'')
                        elif line.startswith('description:'):
                            description = line.split(':', 1)[1].strip().strip('"\'')
                    if name:
                        skills.append((name, description))
                except Exception:
                    continue

        if not skills:
            await ctx.reply(
                "No Agent Skills installed.\n"
                "Use the CLI to install some: `npx skills add [repo]/[project] -a opencode`"
            )
            return

        lines = ["**Installed Agent Skills:**"]
        for name, description in skills:
            lines.append(f"• **{name}**: {description}")
        lines.append("\n*The bot will automatically activate relevant skills when needed.*")
        for part in split_for_discord("\n".join(lines)):
            await ctx.reply(part)

async def setup(bot):
    await bot.add_cog(SkillsCog(bot))