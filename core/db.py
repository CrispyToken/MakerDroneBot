import sqlite3
import aiosqlite
import re
from datetime import datetime, timezone
import discord
from config import DB_PATH

_SCHEMA_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS staff_roles (
        role_id INTEGER PRIMARY KEY, role_name TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS user_profiles (
        user_id INTEGER PRIMARY KEY, username TEXT, display_name TEXT,
        roles TEXT, join_date TEXT, last_seen TEXT,
        message_count INTEGER DEFAULT 0, staff_notes TEXT DEFAULT ''
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS monitored_channels (
        channel_id INTEGER PRIMARY KEY, channel_name TEXT, reason TEXT,
        keywords TEXT, last_message_id INTEGER DEFAULT 0, active INTEGER DEFAULT 1
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS permitted_channels (
        channel_id INTEGER PRIMARY KEY, channel_name TEXT, added_at TEXT,
        window_start_message_id INTEGER DEFAULT 0
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS bot_config (key TEXT PRIMARY KEY, value TEXT)
    """,
)

_MIGRATION_ADD_ACTIVE_COLUMN = "ALTER TABLE monitored_channels ADD COLUMN active INTEGER DEFAULT 1"
_MIGRATION_ADD_WINDOW_START_COLUMN = "ALTER TABLE permitted_channels ADD COLUMN window_start_message_id INTEGER DEFAULT 0"

_SQL_TRACK_USER = """
    INSERT INTO user_profiles (user_id, username, display_name, roles, join_date, last_seen, message_count)
    VALUES (?, ?, ?, ?, ?, ?, 1) ON CONFLICT(user_id) DO UPDATE SET
        username = excluded.username, display_name = excluded.display_name,
        roles = excluded.roles, last_seen = excluded.last_seen, message_count = message_count + 1
"""
_SQL_TRACK_USER_OUTSIDE_GUILD = """
    INSERT INTO user_profiles (user_id, username, display_name, roles, join_date, last_seen, message_count)
    VALUES (?, ?, ?, '', '', ?, 1) ON CONFLICT(user_id) DO UPDATE SET
        username = excluded.username, display_name = excluded.display_name,
        last_seen = excluded.last_seen, message_count = message_count + 1
"""
_SQL_ENSURE_USER = """
    INSERT OR IGNORE INTO user_profiles (user_id, username, display_name, roles, join_date, last_seen, message_count)
    VALUES (?, ?, ?, ?, ?, ?, 0)
"""
_SQL_GET_USER_PROFILE = "SELECT * FROM user_profiles WHERE user_id = ?"
_SQL_GET_USER_BY_QUERY = """
    SELECT * FROM user_profiles 
    WHERE username = ? OR display_name = ? OR user_id = ?
    LIMIT 1
"""
_SQL_GET_CONFIG_VALUE = "SELECT value FROM bot_config WHERE key = ?"
_SQL_SET_CONFIG_VALUE = """
    INSERT INTO bot_config (key, value) VALUES (?, ?)
    ON CONFLICT(key) DO UPDATE SET value = excluded.value
"""
_SQL_GET_MONITORED_CHANNELS = "SELECT * FROM monitored_channels WHERE active = 1"
_SQL_IS_CHANNEL_PERMITTED = "SELECT 1 FROM permitted_channels WHERE channel_id = ?"
_SQL_GET_PERMITTED_CHANNELS = "SELECT * FROM permitted_channels ORDER BY added_at"
_SQL_ADD_PERMITTED_CHANNEL = """
    INSERT INTO permitted_channels (channel_id, channel_name, added_at) VALUES (?, ?, ?)
    ON CONFLICT(channel_id) DO UPDATE SET channel_name = excluded.channel_name
"""
_SQL_REMOVE_PERMITTED_CHANNEL = "DELETE FROM permitted_channels WHERE channel_id = ?"
_SQL_GET_TALK_WINDOW_START = "SELECT window_start_message_id FROM permitted_channels WHERE channel_id = ?"
_SQL_SET_TALK_WINDOW_START = """
    INSERT INTO permitted_channels (channel_id, channel_name, added_at, window_start_message_id)
    VALUES (?, ?, ?, ?)
    ON CONFLICT(channel_id) DO UPDATE SET window_start_message_id = excluded.window_start_message_id
"""
_SQL_GET_STAFF_ROLE_IDS = "SELECT role_id FROM staff_roles"
_SQL_GET_MONITORED_LAST_MESSAGE_ID = "SELECT last_message_id FROM monitored_channels WHERE channel_id = ?"
_SQL_UPSERT_MONITORED_CHANNEL = """
    INSERT INTO monitored_channels (channel_id, channel_name, reason, keywords, last_message_id, active)
    VALUES (?, ?, ?, ?, 0, 1) ON CONFLICT(channel_id) DO
    UPDATE SET channel_name = excluded.channel_name, reason = excluded.reason, keywords = excluded.keywords, active = 1
"""
_SQL_DEACTIVATE_MONITORED_CHANNEL = "UPDATE monitored_channels SET active = 0 WHERE channel_id = ?"
_SQL_SET_MONITORED_LAST_MESSAGE_ID = "UPDATE monitored_channels SET last_message_id = ? WHERE channel_id = ?"
_SQL_APPEND_STAFF_NOTE = """
    UPDATE user_profiles
    SET staff_notes = CASE WHEN staff_notes = '' THEN ? ELSE staff_notes || '\n' || ? END
    WHERE user_id = ?
"""
_SQL_CLEAR_STAFF_NOTES = "UPDATE user_profiles SET staff_notes = '' WHERE user_id = ?"


async def _apply_migrations(db: aiosqlite.Connection) -> None:
    for statement in (_MIGRATION_ADD_ACTIVE_COLUMN, _MIGRATION_ADD_WINDOW_START_COLUMN):
        try:
            await db.execute(statement)
        except sqlite3.OperationalError as error:
            if "duplicate column" not in str(error).lower():
                raise


async def init_db() -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        for statement in _SCHEMA_STATEMENTS:
            await db.execute(statement)
        await _apply_migrations(db)
        await db.commit()


async def track_user(message: discord.Message) -> None:
    if message.author.bot: return
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        if isinstance(message.author, discord.Member):
            roles = ", ".join([r.name for r in message.author.roles if r.name != "@everyone"])
            join_date = message.author.joined_at.isoformat() if message.author.joined_at else ""
            await db.execute(
                _SQL_TRACK_USER,
                (message.author.id, message.author.name, message.author.display_name, roles, join_date, now),
            )
        else:
            await db.execute(
                _SQL_TRACK_USER_OUTSIDE_GUILD,
                (message.author.id, message.author.name, message.author.display_name, now),
            )
        await db.commit()


async def ensure_user_in_db(member: discord.Member) -> None:
    roles = ", ".join([r.name for r in member.roles if r.name != "@everyone"])
    join_date = member.joined_at.isoformat() if member.joined_at else ""
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            _SQL_ENSURE_USER,
            (member.id, member.name, member.display_name, roles, join_date, now),
        )
        await db.commit()


async def get_user_profile(user_id: int) -> dict | None:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(_SQL_GET_USER_PROFILE, (user_id,)) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None

async def get_user_profile_by_query(query: str) -> dict | None:
    query = query.strip()
    mention_match = re.match(r"<@!?(\d+)>", query)
    user_id = int(mention_match.group(1)) if mention_match else (int(query) if query.isdigit() else 0)
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(_SQL_GET_USER_BY_QUERY, (query, query, user_id)) as cursor:
            row = await cursor.fetchone()
            return dict(row) if row else None

async def get_config(key: str) -> str | None:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(_SQL_GET_CONFIG_VALUE, (key,)) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else None


async def set_config(key: str, value: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(_SQL_SET_CONFIG_VALUE, (key, value))
        await db.commit()


async def get_monitored_channels() -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(_SQL_GET_MONITORED_CHANNELS) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]


async def is_channel_permitted(channel_id: int) -> bool:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(_SQL_IS_CHANNEL_PERMITTED, (channel_id,)) as cursor:
            return await cursor.fetchone() is not None


async def get_permitted_channels() -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(_SQL_GET_PERMITTED_CHANNELS) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]


async def add_permitted_channel(channel_id: int, channel_name: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(_SQL_ADD_PERMITTED_CHANNEL, (channel_id, channel_name, now))
        await db.commit()


async def remove_permitted_channel(channel_id: int) -> bool:
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(_SQL_REMOVE_PERMITTED_CHANNEL, (channel_id,))
        await db.commit()
        return cursor.rowcount > 0

async def get_talk_window_start(channel_id: int) -> int:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(_SQL_GET_TALK_WINDOW_START, (channel_id,)) as cursor:
            row = await cursor.fetchone()
            return int(row[0]) if row and row[0] else 0

async def set_talk_window_start(channel_id: int, message_id: int, channel_name: str = "") -> None:
    now = datetime.now(timezone.utc).isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(_SQL_SET_TALK_WINDOW_START, (channel_id, channel_name, now, message_id))
        await db.commit()


async def get_staff_role_ids() -> set[int]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(_SQL_GET_STAFF_ROLE_IDS) as cursor:
            rows = await cursor.fetchall()
            return {row["role_id"] for row in rows}


async def get_monitored_last_message_id(channel_id: int) -> int | None:
    async with aiosqlite.connect(DB_PATH) as db:
        async with db.execute(_SQL_GET_MONITORED_LAST_MESSAGE_ID, (channel_id,)) as cursor:
            row = await cursor.fetchone()
            return row[0] if row else None


async def upsert_monitored_channel(channel_id: int, channel_name: str, reason: str, keywords: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(_SQL_UPSERT_MONITORED_CHANNEL, (channel_id, channel_name, reason, keywords))
        await db.commit()


async def deactivate_monitored_channel(channel_id: int) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(_SQL_DEACTIVATE_MONITORED_CHANNEL, (channel_id,))
        await db.commit()


async def set_monitored_last_message_ids(updates: dict[int, int]) -> None:
    if not updates:
        return
    async with aiosqlite.connect(DB_PATH) as db:
        for channel_id, last_message_id in updates.items():
            await db.execute(_SQL_SET_MONITORED_LAST_MESSAGE_ID, (last_message_id, channel_id))
        await db.commit()


async def append_staff_note(user_id: int, note: str) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(_SQL_APPEND_STAFF_NOTE, (note, note, user_id))
        await db.commit()


async def clear_staff_notes(user_id: int) -> None:
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(_SQL_CLEAR_STAFF_NOTES, (user_id,))
        await db.commit()