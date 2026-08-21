import aiosqlite
from datetime import datetime, timezone
import discord
from config import DB_PATH

async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id INTEGER PRIMARY KEY, username TEXT, display_name TEXT,
                roles TEXT, join_date TEXT, last_seen TEXT,
                message_count INTEGER DEFAULT 0, staff_notes TEXT DEFAULT ''
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS monitored_channels (
                channel_id INTEGER PRIMARY KEY, channel_name TEXT, reason TEXT,
                keywords TEXT, last_message_id INTEGER DEFAULT 0
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS bot_config (key TEXT PRIMARY KEY, value TEXT)
        """)
        await db.commit()

async def track_user(message: discord.Message):
    if message.author.bot: return
    roles = ", ".join([r.name for r in message.author.roles if r.name != "@everyone"])
    join_date = message.author.joined_at.isoformat() if message.author.joined_at else ""
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            INSERT INTO user_profiles (user_id, username, display_name, roles, join_date, last_seen, message_count)
            VALUES (?, ?, ?, ?, ?, ?, 1) ON CONFLICT(user_id) DO UPDATE SET
            username = excluded.username, display_name = excluded.display_name,
            roles = excluded.roles, last_seen = excluded.last_seen, message_count = message_count + 1
        """, (message.author.id, message.author.name, message.author.display_name, roles, join_date, now))
        await db.commit()

async def ensure_user_in_db(member: discord.Member):
    roles = ", ".join([r.name for r in member.roles if r.name != "@everyone"])
    join_date = member.joined_at.isoformat() if member.joined_at else ""
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            INSERT OR IGNORE INTO user_profiles (user_id, username, display_name, roles, join_date, last_seen, message_count)
            VALUES (?, ?, ?, ?, ?, ?, 0)
        """, (member.id, member.name, member.display_name, roles, join_date, now))
        await db.commit()

async def get_user_profile(user_id: int) -> dict | None:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM user_profiles WHERE user_id = ?", (user_id,)) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None

async def get_config(key: str) -> str | None:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute("SELECT value FROM bot_config WHERE key = ?", (key,)) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else None

async def set_config(key: str, value: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            INSERT INTO bot_config (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """, (key, value))
        await db.commit()

async def get_monitored_channels() -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute("SELECT * FROM monitored_channels") as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]