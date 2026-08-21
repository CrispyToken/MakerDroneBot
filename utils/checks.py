from discord.ext import commands

def is_staff():
    async def predicate(ctx: commands.Context) -> bool:
        if ctx.guild is None: return True
        return ctx.author.guild_permissions.manage_messages
    return commands.check(predicate)