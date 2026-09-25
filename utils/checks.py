from discord.ext import commands
from core.db import get_staff_role_ids


def is_staff():
    async def predicate(ctx: commands.Context) -> bool:
        if ctx.guild is None:
            return False  # Staff commands are disabled in DMs

        staff_roles = await get_staff_role_ids()
        if not staff_roles:
            return False  # No roles configured in DB = no staff access

        user_role_ids = {role.id for role in ctx.author.roles}
        return bool(user_role_ids & staff_roles)

    return commands.check(predicate)