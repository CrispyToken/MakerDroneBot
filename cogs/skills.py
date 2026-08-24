# cogs/skills.py
import discord
from discord.ext import commands
import services.skills as skills_module
from utils.formatting import split_for_discord

class SkillsCog(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command(name="skills", help="List all installed Agent Skills.")
    async def list_skills(self, ctx: commands.Context):
        if not skills_module.skill_manager or not skills_module.skill_manager.skills:
            await ctx.reply(
                "No Agent Skills installed.\n"
                "Use the CLI to install some: `npx skills add Leonxlnx/unlazy -a opencode`"
            )
            return
        
        lines = ["**Installed Agent Skills:**"]
        for skill in skills_module.skill_manager.skills.values():
            lines.append(f"• **{skill.name}**: {skill.description}")
        
        lines.append("\n*The bot will automatically activate relevant skills, or you can ask it to 'use the [name] skill'.*")
        
        for part in split_for_discord("\n".join(lines)):
            await ctx.reply(part)

    @commands.command(name="reloadskills", help="Reload SKILL.md files from disk.")
    @commands.has_permissions(manage_messages=True)
    async def reload_skills(self, ctx: commands.Context):
        if skills_module.skill_manager:
            skills_module.skill_manager.reload()
            await ctx.reply(f"Reloaded {len(skills_module.skill_manager.skills)} skills from disk.")
        else:
            await ctx.reply("Skill manager not initialized.")

async def setup(bot):
    await bot.add_cog(SkillsCog(bot))