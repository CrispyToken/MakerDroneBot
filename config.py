import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Discord
TOKEN = os.getenv("DISCORD_TOKEN", "").strip()
COMMAND_PREFIX = os.getenv("COMMAND_PREFIX", "!")

# Cognee
COGNEE_CHUNK_SIZE = int(os.getenv("COGNEE_CHUNK_SIZE", "1200"))
COGNEE_CHUNK_OVERLAP = int(os.getenv("COGNEE_CHUNK_OVERLAP", "150"))

# SearXNG
SEARXNG_URL = os.getenv("SEARXNG_URL", "http://localhost:8080").rstrip("/")

# System Prompts
SYSTEM_PROMPT_FILE = Path(os.getenv("SYSTEM_PROMPT_FILE", "system_prompt.txt")).resolve()
MONITOR_PROMPT_FILE = Path(os.getenv("MONITOR_PROMPT_FILE", "monitoring_prompt.txt")).resolve()
SERVER_RULES_FILE = Path(os.getenv("SERVER_RULES_FILE", "server_rules.txt")).resolve()

# LM Studio
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234").rstrip("/")
if not LMSTUDIO_BASE_URL.endswith("/v1"):
    LMSTUDIO_BASE_URL += "/v1"
LMSTUDIO_API_KEY = os.getenv("LMSTUDIO_API_KEY", "").strip()
LMSTUDIO_CHAT_MODEL = os.getenv("LMSTUDIO_CHAT_MODEL", "local-model")

# Monitoring
MONITOR_INTERVAL_MINUTES = int(os.getenv("MONITOR_INTERVAL_MINUTES", "180"))
MONITOR_MAX_MESSAGES_PER_CHANNEL = int(os.getenv("MONITOR_MAX_MESSAGES_PER_CHANNEL", "50"))

# Models
FAST_MODEL_ID = os.getenv("FAST_MODEL_ID", LMSTUDIO_CHAT_MODEL).strip()
DEEP_MODEL_ID = os.getenv("DEEP_MODEL_ID", FAST_MODEL_ID).strip()
MODEL_SWITCH_MODE = os.getenv("MODEL_SWITCH_MODE", "command").lower()
FAST_MODEL_LOAD_COMMAND = os.getenv("FAST_MODEL_LOAD_COMMAND", "").strip()
DEEP_MODEL_LOAD_COMMAND = os.getenv("DEEP_MODEL_LOAD_COMMAND", "").strip()
MODEL_SWITCH_TIMEOUT = int(os.getenv("MODEL_SWITCH_TIMEOUT", "180"))
MODEL_SWITCH_SETTLE_SECONDS = float(os.getenv("MODEL_SWITCH_SETTLE_SECONDS", "3"))

# Images
MAX_IMAGE_ATTACHMENTS = int(os.getenv("MAX_IMAGE_ATTACHMENTS", "3"))
MAX_IMAGE_MB = int(os.getenv("MAX_IMAGE_MB", "10"))
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
TARGET_IMAGE_PIXELS = int(os.getenv("TARGET_IMAGE_PIXELS", "1000000"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "85"))

# Text/Docs
MAX_TEXT_ATTACHMENTS = int(os.getenv("MAX_TEXT_ATTACHMENTS", "3"))
MAX_TEXT_ATTACHMENT_CHARS = int(os.getenv("MAX_TEXT_ATTACHMENT_CHARS", "6000"))
MAX_TOTAL_TEXT_ATTACHMENT_CHARS = int(os.getenv("MAX_TOTAL_TEXT_ATTACHMENT_CHARS", "15000"))
MAX_TEXT_ATTACHMENT_MB = int(os.getenv("MAX_TEXT_ATTACHMENT_MB", "20"))

# Ingest
INGEST_DIR = Path(os.getenv("INGEST_DIR", "ingest")).resolve()
INGEST_DIR.mkdir(parents=True, exist_ok=True)
TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".log", ".csv", ".json"}
INGEST_EXTENSIONS = TEXT_EXTENSIONS.union({".pdf", ".docx", ".html", ".htm", ".epub"})
ATTACHMENT_TEXT_EXTENSIONS = INGEST_EXTENSIONS.union({".py", ".yaml", ".yml", ".xml", ".toml", ".ini", ".cfg", ".conf"})

# Storage
DATA_DIR = Path(os.getenv("DATA_DIR", "data")).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DATA_DIR / "memory.db"
HASH_RECORD_PATH = DATA_DIR / "ingest_hashes.json"
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "180"))

# Discord Limits
DISCORD_CHAR_LIMIT = max(500, min(int(os.getenv("DISCORD_CHAR_LIMIT", "2000")), 2000))
DISCORD_MAX_SPLIT_CHARS = max(DISCORD_CHAR_LIMIT, min(int(os.getenv("DISCORD_MAX_SPLIT_CHARS", "4000")), DISCORD_CHAR_LIMIT * 2))