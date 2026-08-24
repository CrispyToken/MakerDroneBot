"""
Scoped reasoning ("thinking") control for the local llama.cpp server.

The Ling / Bailing-MoE model reasons by default. That's what we want for the
Discord chat agent, but it makes Cognee's internal LLM operations (entity
extraction, summarisation, graph construction, recall) painfully slow.

llama.cpp's server honours a per-request ``chat_template_kwargs`` body field and
the Bailing chat template's thinking switch is ``enable_thinking``.

Design
------
* Reasoning is DISABLED by default: the OpenAI SDK is monkey-patched so every
  chat-completion request gets
  ``extra_body={"chat_template_kwargs": {"enable_thinking": False}}`` injected.
* The chat agent OPTS BACK IN by setting the ``chat_reasoning_enabled``
  contextvar (via :func:`chat_reasoning`) around ``agent.run``.
* Cognee operations are scheduled onto the dedicated background loop in
  ``core.memory`` with ``asyncio.run_coroutine_threadsafe``, which starts from a
  fresh context, so they never see the opt-in flag and always run with reasoning
  disabled. Same for the direct ``cognee.*`` awaits in ``cogs/staff.py``.
"""

import contextvars
import logging
from contextlib import asynccontextmanager

log = logging.getLogger("rag-bot")

# True => this request may reason (chat agent). Default False => reasoning off.
chat_reasoning_enabled: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "chat_reasoning_enabled", default=False
)

_TEMPLATE_KWARGS_KEY = "chat_template_kwargs"
_installed = False


@asynccontextmanager
async def chat_reasoning():
    """Opt the current task into reasoning (wrap the chat agent run with this)."""
    token = chat_reasoning_enabled.set(True)
    try:
        yield
    finally:
        chat_reasoning_enabled.reset(token)


def _inject_no_think(kwargs: dict) -> dict:
    """Force ``enable_thinking`` off unless the call is flagged as chat reasoning."""
    if chat_reasoning_enabled.get():
        # Chat path: leave the request alone so the model keeps default thinking.
        return kwargs

    extra_body = kwargs.get("extra_body")
    extra_body = dict(extra_body) if extra_body else {}

    template_kwargs = dict(extra_body.get(_TEMPLATE_KWARGS_KEY) or {})
    # Only force it off if the caller hasn't explicitly chosen a value.
    template_kwargs.setdefault("enable_thinking", False)

    extra_body[_TEMPLATE_KWARGS_KEY] = template_kwargs
    kwargs["extra_body"] = extra_body
    return kwargs


def install_reasoning_patch() -> None:
    """Monkey-patch the OpenAI SDK. Call once at startup before any LLM request."""
    global _installed
    if _installed:
        return

    try:
        from openai.resources.chat.completions import AsyncCompletions, Completions
    except Exception:
        log.exception("Could not import OpenAI SDK; reasoning control NOT installed.")
        return

    async_create = AsyncCompletions.create

    async def _patched_async_create(self, *args, **kwargs):
        return await async_create(self, *args, **_inject_no_think(kwargs))

    AsyncCompletions.create = _patched_async_create

    sync_create = Completions.create

    def _patched_sync_create(self, *args, **kwargs):
        return sync_create(self, *args, **_inject_no_think(kwargs))

    Completions.create = _patched_sync_create

    _installed = True
    log.info("Reasoning control installed: thinking OFF by default, chat agent opts in.")