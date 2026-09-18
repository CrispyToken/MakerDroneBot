"""
Global LLM lock and active-task tracking.

One asyncio.Lock shared by every subsystem that occupies the local
inference engine: chat turns, monitoring evaluations, and document
ingestion. Whoever acquires it keeps it until fully done, then releases.

The lock is only ever acquired on the main Discord event loop. LightRAG /
memory work runs on the background loop *under* a lock already held by one
of these entry points, and never acquires the lock itself.
"""

import asyncio
from contextlib import asynccontextmanager

llm_lock = asyncio.Lock()

_active_task: asyncio.Task | None = None
_active_label: str | None = None


@asynccontextmanager
async def track_llm_task(label: str):
    """Register the current task as the active LLM task for the block."""
    global _active_task, _active_label
    prev_task, prev_label = _active_task, _active_label
    _active_task = asyncio.current_task()
    _active_label = label
    try:
        yield
    finally:
        _active_task, _active_label = prev_task, prev_label


def interrupt_active_llm() -> str | None:
    """Cancel the currently tracked LLM task, if any.

    Returns the interrupted task's label, or None if nothing was active.
    The inference server and loaded model are not touched.
    """
    if _active_task is None or _active_task.done():
        return None
    label = _active_label or "unknown task"
    _active_task.cancel()
    return label