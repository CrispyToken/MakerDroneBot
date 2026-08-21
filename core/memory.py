import asyncio
import threading
import json
import hashlib
import logging
from pathlib import Path
import cognee
from config import HASH_RECORD_PATH

log = logging.getLogger("rag-bot")

# ---------------------------------------------------------------------------
# Persistent background event loop
# ---------------------------------------------------------------------------
# Cognee (via litellm + aiohttp) spawns background tasks and holds open
# connections tied to the event loop they were created on. The old approach
# created a NEW event loop for every operation and closed it right after,
# which orphaned litellm's LoggingWorker and left aiohttp sessions unclosed
# ("Event loop is closed" / "Unclosed client session").
#
# Instead we run ONE long-lived loop on a dedicated daemon thread and
# schedule every Cognee coroutine onto it. Tasks and connections are reused
# across operations and never torn down mid-flight.
# ---------------------------------------------------------------------------

_bg_loop: asyncio.AbstractEventLoop | None = None
_bg_lock = threading.Lock()


def _get_bg_loop() -> asyncio.AbstractEventLoop:
    global _bg_loop
    with _bg_lock:
        if _bg_loop is None or not _bg_loop.is_running():
            _bg_loop = asyncio.new_event_loop()
            threading.Thread(
                target=_bg_loop.run_forever,
                name="cognee-bg-loop",
                daemon=True,
            ).start()
        return _bg_loop


async def cognee_in_background(coro_func, *args, **kwargs):
    """Run a coroutine function on the persistent background loop and await
    its result without blocking the Discord gateway loop."""
    loop = _get_bg_loop()
    future = asyncio.run_coroutine_threadsafe(coro_func(*args, **kwargs), loop)
    return await asyncio.wrap_future(future)


async def _close_graph_engine():
    """Close the Cognee graph engine to release the DB file lock.
    Must run on the background loop (where the engine lives)."""
    try:
        from cognee.infrastructure.databases.graph.get_graph_engine import get_graph_engine
        engine = await get_graph_engine()
        if hasattr(engine, "close"):
            result = engine.close()
            if asyncio.iscoroutine(result):
                await result
        if hasattr(engine, "db") and hasattr(engine.db, "close"):
            result = engine.db.close()
            if asyncio.iscoroutine(result):
                await result
        log.debug("Cognee graph engine closed. File lock released.")
    except Exception as e:
        log.warning("Failed to force-close Cognee engine: %s", e)


async def release_cognee_lock():
    """Public helper. Safe to call from the Discord loop; the actual close
    is scheduled onto the background loop."""
    await cognee_in_background(_close_graph_engine)


async def cognee_recall(query: str, datasets: list[str] | None = None) -> list:
    """Run cognee.recall on the background loop and release the lock after."""
    async def _do_recall():
        try:
            kwargs = {}
            if datasets:
                kwargs["datasets"] = datasets
            return await cognee.recall(query, **kwargs)
        finally:
            await _close_graph_engine()
    return await cognee_in_background(_do_recall)


async def warmup_cognee():
    """Pre-load the embedding model + vector engine at startup so the first
    real recall doesn't pay the download/init cost."""
    try:
        log.info("Warming up Cognee (pre-loading embedding model and vector engine)...")
        await cognee_in_background(cognee.recall, "warmup initialization")
        log.info("Cognee warmup complete.")
    except Exception as e:
        log.warning("Cognee warmup failed (non-critical): %s", e)


def load_ingest_hashes() -> dict[str, str]:
    if HASH_RECORD_PATH.exists():
        try:
            return json.loads(HASH_RECORD_PATH.read_text(encoding="utf-8"))
        except Exception:
            log.exception("Failed to read ingest hash record, starting fresh.")
            return {}
    return {}


def save_ingest_hashes(records: dict[str, str]) -> None:
    HASH_RECORD_PATH.write_text(
        json.dumps(records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def compute_file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    hasher.update(path.read_bytes())
    return hasher.hexdigest()