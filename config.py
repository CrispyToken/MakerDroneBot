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
SYSTEM_PROMPT_FILE = Path(os.getenv("SYSTEM_PROMPT_FILE", "prompt/system_prompt.txt")).resolve()
MONITOR_PROMPT_FILE = Path(os.getenv("MONITOR_PROMPT_FILE", "prompt/monitoring_prompt.txt")).resolve()
SERVER_RULES_FILE = Path(os.getenv("SERVER_RULES_FILE", "prompt/server_rules.txt")).resolve()

# Standard directories where 'npx skills' installs SKILL.md files for various agents
SKILLS_DIRS = [
    Path(os.getenv("SKILLS_DIR", "skills")).resolve(),
    Path(".agents/skills").resolve(),
    Path(".claude/skills").resolve(),
    Path(".cursor/skills").resolve(),
]

# LLM / Inference Backend
# We read LLM_ENDPOINT and LLM_MODEL (which Cognee/LiteLLM require).
LLM_BASE_URL = os.getenv("LLM_ENDPOINT", "http://127.0.0.1:1234/v1").rstrip("/")
if not LLM_BASE_URL.endswith("/v1"):
    LLM_BASE_URL += "/v1"
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()

# LiteLLM (Cognee) requires the "openai/" prefix in LLM_MODEL.
# PydanticAI (the bot) just wants the raw model ID. We strip it here.
raw_model = os.getenv("LLM_MODEL", "local-model").strip()
LLM_MODEL_ID = raw_model.split("/", 1)[-1] if "/" in raw_model else raw_model

# Monitoring
MONITOR_INTERVAL_MINUTES = int(os.getenv("MONITOR_INTERVAL_MINUTES", "180"))
MONITOR_MAX_MESSAGES_PER_CHANNEL = int(os.getenv("MONITOR_MAX_MESSAGES_PER_CHANNEL", "50"))

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