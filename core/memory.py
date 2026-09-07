import asyncio
import threading
import json
import hashlib
import logging
from pathlib import Path

import numpy as np

from config import (
    HASH_RECORD_PATH, LLM_BASE_URL, LLM_API_KEY, LLM_MODEL_ID,
    LIGHTRAG_DIR, LIGHTRAG_EMBED_MODEL, LIGHTRAG_EMBED_DIM,
    LIGHTRAG_CHUNK_SIZE, LIGHTRAG_CHUNK_OVERLAP, LIGHTRAG_QUERY_MAX_TOKENS,
    EXTRACT_LLM_TIMEOUT, LLM_TIMEOUT,
)

log = logging.getLogger("rag-bot")

KNOWLEDGE_WORKSPACE = "knowledge"
DYNAMIC_WORKSPACE = "dynamic"

# ---------------------------------------------------------------------------
# Persistent background event loop
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
                name="memory-bg-loop",
                daemon=True,
            ).start()
    return _bg_loop


async def memory_in_background(coro_func, *args, **kwargs):
    """Run a coroutine function on the persistent background loop and await
    its result without blocking the Discord gateway loop."""
    loop = _get_bg_loop()
    future = asyncio.run_coroutine_threadsafe(coro_func(*args, **kwargs), loop)
    return await asyncio.wrap_future(future)


# ---------------------------------------------------------------------------
# LLM + embedding providers
# ---------------------------------------------------------------------------
_embed_model = None
_embed_lock = threading.Lock()


async def _llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
    """Route LightRAG's LLM calls to the local llama-server (OpenAI API)."""
    from lightrag.llm.openai import openai_complete_if_cache
    default_timeout = max(EXTRACT_LLM_TIMEOUT, LLM_TIMEOUT)
    timeout = kwargs.pop("timeout", default_timeout)
    return await openai_complete_if_cache(
        LLM_MODEL_ID,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages or [],
        api_key=LLM_API_KEY or "no-key",
        base_url=LLM_BASE_URL,
        timeout=timeout,
        **kwargs,
    )


def _get_embed_model():
    global _embed_model
    with _embed_lock:
        if _embed_model is None:
            from fastembed import TextEmbedding
            log.info("Loading embedding model: %s", LIGHTRAG_EMBED_MODEL)
            _embed_model = TextEmbedding(model_name=LIGHTRAG_EMBED_MODEL)
    return _embed_model


async def _embedding_func(texts: list[str]) -> np.ndarray:
    def _embed():
        model = _get_embed_model()
        return np.array(list(model.embed(texts)))
    return await asyncio.to_thread(_embed)


# ---------------------------------------------------------------------------
# LightRAG instances
# ---------------------------------------------------------------------------
_rags: dict = {}
_init_lock = asyncio.Lock()


async def _build_rag(workspace: str):
    from lightrag import LightRAG
    from lightrag.utils import EmbeddingFunc

    working_dir = LIGHTRAG_DIR / workspace
    working_dir.mkdir(parents=True, exist_ok=True)

    rag = LightRAG(
        working_dir=str(working_dir),
        workspace=workspace,
        llm_model_func=_llm_model_func,
        llm_model_name=LLM_MODEL_ID,
        embedding_func=EmbeddingFunc(
            embedding_dim=LIGHTRAG_EMBED_DIM,
            max_token_size=8192,
            model_name=LIGHTRAG_EMBED_MODEL,
            func=_embedding_func,
        ),
        chunk_token_size=LIGHTRAG_CHUNK_SIZE,
        chunk_overlap_token_size=LIGHTRAG_CHUNK_OVERLAP,
        enable_llm_cache=True,
        enable_llm_cache_for_entity_extract=True,
    )
    await rag.initialize_storages()
    log.info("LightRAG workspace '%s' initialized at %s", workspace, working_dir)
    return rag


async def _get_rag(workspace: str):
    async with _init_lock:
        if workspace not in _rags:
            _rags[workspace] = await _build_rag(workspace)
    return _rags[workspace]


def _query_param():
    from lightrag import QueryParam
    return QueryParam(
        mode="hybrid",
        only_need_context=True,
        enable_rerank=False,
        max_total_tokens=LIGHTRAG_QUERY_MAX_TOKENS,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
async def memory_remember(fact: str) -> None:
    """Store one dynamic memory fact (entity extraction runs in background)."""
    async def _do():
        rag = await _get_rag(DYNAMIC_WORKSPACE)
        await rag.ainsert(fact)
    await memory_in_background(_do)


async def memory_recall_dynamic(query: str) -> str:
    """Recall retrieved context from dynamic memories (no answer synthesis)."""
    async def _do():
        rag = await _get_rag(DYNAMIC_WORKSPACE)
        return await rag.aquery(query, param=_query_param())
    return await memory_in_background(_do)


async def memory_recall_knowledge(query: str) -> str:
    """Recall retrieved context from ingested documents (no answer synthesis)."""
    async def _do():
        rag = await _get_rag(KNOWLEDGE_WORKSPACE)
        return await rag.aquery(query, param=_query_param())
    return await memory_in_background(_do)


async def ingest_document(text: str, doc_id: str) -> None:
    """Ingest one document into the knowledge workspace under a stable doc ID."""
    async def _do():
        rag = await _get_rag(KNOWLEDGE_WORKSPACE)
        await rag.ainsert(text, ids=[doc_id])
    await memory_in_background(_do)


async def forget_document(doc_id: str) -> None:
    """Remove one document (chunks, unique entities/relations, vectors)."""
    async def _do():
        rag = await _get_rag(KNOWLEDGE_WORKSPACE)
        await rag.adelete_by_doc_id(doc_id)
    await memory_in_background(_do)


async def warmup_memory():
    """Initialize both workspaces and preload the embedding model."""
    try:
        log.info("Warming up memory backends (LightRAG + embeddings)...")
        async def _do():
            await _get_rag(KNOWLEDGE_WORKSPACE)
            await _get_rag(DYNAMIC_WORKSPACE)
            await _embedding_func(["warmup"])
        await memory_in_background(_do)
        log.info("Memory backends ready.")
    except Exception as e:
        log.warning("Memory warmup failed (non-critical): %s", e)


async def shutdown_memory():
    """Finalize storages and stop the background loop. Call on bot shutdown."""
    global _bg_loop
    async def _do():
        for rag in _rags.values():
            try:
                await rag.finalize_storages()
            except Exception as e:
                log.warning("finalize_storages failed: %s", e)
    try:
        await memory_in_background(_do)
    except Exception:
        pass
    with _bg_lock:
        if _bg_loop is not None:
            _bg_loop.call_soon_threadsafe(_bg_loop.stop)
            _bg_loop = None
    log.info("Memory backends shut down.")


# ---------------------------------------------------------------------------
# Ingest hash helpers (unchanged)
# ---------------------------------------------------------------------------
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