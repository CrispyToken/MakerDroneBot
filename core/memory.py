import asyncio
import threading
import json
import hashlib
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any
import numpy as np
from config import (
    HASH_RECORD_PATH, LLM_BASE_URL, LLM_API_KEY, LLM_MODEL_ID,
    LIGHTRAG_DIR, LIGHTRAG_EMBED_MODEL, LIGHTRAG_EMBED_DIM,
    LIGHTRAG_CHUNK_SIZE, LIGHTRAG_CHUNK_OVERLAP, LIGHTRAG_QUERY_MAX_TOKENS,
    EXTRACT_LLM_TIMEOUT, LLM_TIMEOUT, LLM_MAX_EXTRACT_OUTPUT_TOKENS,
)

log = logging.getLogger("rag-bot")

KNOWLEDGE_WORKSPACE = "knowledge"
DYNAMIC_WORKSPACE = "dynamic"
if TYPE_CHECKING:
    from fastembed import TextEmbedding
    from lightrag import LightRAG, QueryParam


class MemoryManager:
    """Encapsulates all memory-backend state: the persistent background event
    loop, the embedding model singleton, and the LightRAG workspace registry.
    All LightRAG / embedding work runs on the background loop so it never
    blocks the Discord gateway loop."""

    def __init__(self) -> None:
        self._bg_loop: asyncio.AbstractEventLoop | None = None
        self._bg_lock = threading.Lock()
        self._bg_active_task: asyncio.Task | None = None

        self._embed_model = None
        self._embed_lock = threading.Lock()
        self._embed_call_lock = threading.Lock()

        self._rags: dict = {}
        self._init_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Persistent background event loop
    # ------------------------------------------------------------------
    def _get_bg_loop(self) -> asyncio.AbstractEventLoop:
        with self._bg_lock:
            if self._bg_loop is None or not self._bg_loop.is_running():
                self._bg_loop = asyncio.new_event_loop()
                threading.Thread(
                    target=self._bg_loop.run_forever,
                    name="memory-bg-loop",
                    daemon=True,
                ).start()
            return self._bg_loop

    async def _run_in_background(self, coro_func: Callable[..., Awaitable[Any]], *args, **kwargs) -> Any:
        """Run a coroutine function on the persistent background loop and await
        its result without blocking the Discord gateway loop."""
        loop = self._get_bg_loop()

        async def _wrapper():
            self._bg_active_task = asyncio.current_task()
            try:
                return await coro_func(*args, **kwargs)
            finally:
                self._bg_active_task = None

        future = asyncio.run_coroutine_threadsafe(_wrapper(), loop)
        try:
            return await asyncio.wrap_future(future)
        except asyncio.CancelledError:
            bg_loop = self._bg_loop
            if bg_loop is not None:
                bg_loop.call_soon_threadsafe(
                    lambda: self._bg_active_task.cancel() if self._bg_active_task is not None else None
                )
            raise

    # ------------------------------------------------------------------
    # LLM + embedding providers
    # ------------------------------------------------------------------
    async def _llm_model_func(self, prompt, system_prompt=None, history_messages=None, **kwargs):
        """Route LightRAG's LLM calls to the local llama-server (OpenAI API)."""
        from lightrag.llm.openai import openai_complete_if_cache

        default_timeout = max(EXTRACT_LLM_TIMEOUT, LLM_TIMEOUT)
        timeout = kwargs.pop("timeout", default_timeout)
        kwargs.setdefault("max_tokens", LLM_MAX_EXTRACT_OUTPUT_TOKENS)

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

    def _get_embed_model(self) -> "TextEmbedding":
        with self._embed_lock:
            if self._embed_model is None:
                from fastembed import TextEmbedding
                log.info("Loading embedding model: %s", LIGHTRAG_EMBED_MODEL)
                self._embed_model = TextEmbedding(model_name=LIGHTRAG_EMBED_MODEL)
            return self._embed_model

    async def _embedding_func(self, texts: list[str]) -> np.ndarray:
        def _embed():
            model = self._get_embed_model()
            with self._embed_call_lock:
                return np.array(list(model.embed(texts)))

        return await asyncio.to_thread(_embed)

    # ------------------------------------------------------------------
    # LightRAG instances
    # ------------------------------------------------------------------
    async def _build_rag(self, workspace: str) -> "LightRAG":
        from lightrag import LightRAG
        from lightrag.utils import EmbeddingFunc

        async def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
            return await self._llm_model_func(prompt, system_prompt, history_messages, **kwargs)

        async def embedding_func(texts: list[str]) -> np.ndarray:
            return await self._embedding_func(texts)

        working_dir = LIGHTRAG_DIR / workspace
        working_dir.mkdir(parents=True, exist_ok=True)

        rag = LightRAG(
            working_dir=str(working_dir),
            workspace=workspace,
            llm_model_func=llm_model_func,
            llm_model_name=LLM_MODEL_ID,
            embedding_func=EmbeddingFunc(
                embedding_dim=LIGHTRAG_EMBED_DIM,
                max_token_size=8192,
                model_name=LIGHTRAG_EMBED_MODEL,
                func=embedding_func,
            ),
            chunk_token_size=LIGHTRAG_CHUNK_SIZE,
            chunk_overlap_token_size=LIGHTRAG_CHUNK_OVERLAP,
            enable_llm_cache=True,
            enable_llm_cache_for_entity_extract=True,
        )
        await rag.initialize_storages()
        log.info("LightRAG workspace '%s' initialized at %s", workspace, working_dir)
        return rag

    async def _get_rag(self, workspace: str):
        async with self._init_lock:
            if workspace not in self._rags:
                self._rags[workspace] = await self._build_rag(workspace)
            return self._rags[workspace]

    def _query_param(self) -> "QueryParam":
        from lightrag import QueryParam
        return QueryParam(
            mode="hybrid",
            only_need_context=True,
            enable_rerank=False,
            max_total_tokens=LIGHTRAG_QUERY_MAX_TOKENS,
        )

    # ------------------------------------------------------------------
    # Operations
    # ------------------------------------------------------------------
    async def remember(self, fact: str) -> None:
        """Store one dynamic memory fact (entity extraction runs in background)."""
        async def _do():
            rag = await self._get_rag(DYNAMIC_WORKSPACE)
            await rag.ainsert(fact)

        await self._run_in_background(_do)

    async def recall_dynamic(self, query: str) -> str:
        """Recall retrieved context from dynamic memories (no answer synthesis)."""
        async def _do():
            rag = await self._get_rag(DYNAMIC_WORKSPACE)
            return await rag.aquery(query, param=self._query_param())

        return await self._run_in_background(_do)

    async def recall_knowledge(self, query: str) -> str:
        """Recall retrieved context from ingested documents (no answer synthesis)."""
        async def _do():
            rag = await self._get_rag(KNOWLEDGE_WORKSPACE)
            return await rag.aquery(query, param=self._query_param())

        return await self._run_in_background(_do)

    async def ingest(self, text: str, doc_id: str) -> None:
        """Ingest one document into the knowledge workspace under a stable doc ID."""
        async def _do():
            rag = await self._get_rag(KNOWLEDGE_WORKSPACE)
            await rag.ainsert(text, ids=[doc_id])

        await self._run_in_background(_do)

    async def forget(self, doc_id: str) -> None:
        """Remove one document (chunks, unique entities/relations, vectors)."""
        async def _do():
            rag = await self._get_rag(KNOWLEDGE_WORKSPACE)
            await rag.adelete_by_doc_id(doc_id)

        await self._run_in_background(_do)

    async def warmup(self) -> None:
        """Initialize both workspaces and preload the embedding model."""
        try:
            log.info("Warming up memory backends (LightRAG + embeddings)...")

            async def _do():
                await self._get_rag(KNOWLEDGE_WORKSPACE)
                await self._get_rag(DYNAMIC_WORKSPACE)
                await self._embedding_func(["warmup"])

            await self._run_in_background(_do)
            log.info("Memory backends ready.")
        except Exception:
            log.warning("Memory warmup failed (non-critical)", exc_info=True)

    async def shutdown(self) -> None:
        """Finalize storages and stop the background loop. Call on bot shutdown."""
        async def _do():
            for rag in self._rags.values():
                try:
                    await rag.finalize_storages()
                except Exception as e:
                    log.warning("finalize_storages failed: %s", e)

        try:
            await self._run_in_background(_do)
        except Exception:
            pass

        with self._bg_lock:
            if self._bg_loop is not None:
                self._bg_loop.call_soon_threadsafe(self._bg_loop.stop)
                self._bg_loop = None

        log.info("Memory backends shut down.")


_memory_manager = MemoryManager()


# ---------------------------------------------------------------------------
# Public API — thin facade preserving the existing call signatures.
# ---------------------------------------------------------------------------
async def memory_remember(fact: str) -> None:
    await _memory_manager.remember(fact)

async def memory_recall_dynamic(query: str) -> str:
    return await _memory_manager.recall_dynamic(query)

async def memory_recall_knowledge(query: str) -> str:
    return await _memory_manager.recall_knowledge(query)

async def ingest_document(text: str, doc_id: str) -> None:
    await _memory_manager.ingest(text, doc_id)

async def forget_document(doc_id: str) -> None:
    await _memory_manager.forget(doc_id)

async def warmup_memory() -> None:
    await _memory_manager.warmup()

async def shutdown_memory() -> None:
    await _memory_manager.shutdown()


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