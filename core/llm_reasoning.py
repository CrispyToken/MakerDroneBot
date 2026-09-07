import contextvars
import logging
from contextlib import asynccontextmanager
from core.console import print_completion

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


def _apply_reasoning_policy(kwargs: dict) -> dict:
    """Force the model's thinking switch explicitly on every request.

    Chat path (contextvar True)  -> enable_thinking: True
    Everything else (memory etc) -> enable_thinking: False

    Explicit on BOTH paths so behaviour never depends on the server-level
    --reasoning default, and the active switch is verifiable in traffic.
    """
    extra_body = kwargs.get("extra_body")
    extra_body = dict(extra_body) if extra_body else {}
    template_kwargs = dict(extra_body.get(_TEMPLATE_KWARGS_KEY) or {})
    template_kwargs["enable_thinking"] = bool(chat_reasoning_enabled.get())
    extra_body[_TEMPLATE_KWARGS_KEY] = template_kwargs
    kwargs["extra_body"] = extra_body
    log.debug("Reasoning policy: enable_thinking=%s", template_kwargs["enable_thinking"])
    return kwargs

def _log_completion_to_console(response) -> None:
    """Print a finished chat completion as structured console blocks."""
    try:
        msg = response.choices[0].message
    except Exception:
        return
    reasoning = getattr(msg, "reasoning_content", None)
    if reasoning is None:
        reasoning = (getattr(msg, "model_extra", None) or {}).get("reasoning_content")
    tool_calls = []
    for tc in (getattr(msg, "tool_calls", None) or []):
        fn = getattr(tc, "function", None)
        if fn is not None:
            tool_calls.append((getattr(fn, "name", "?"), getattr(fn, "arguments", "")))
    print_completion(reasoning or "", getattr(msg, "content", None) or "", tool_calls, source="chat")

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
        kwargs = _apply_reasoning_policy(kwargs)
        response = await async_create(self, *args, **kwargs)
        if chat_reasoning_enabled.get() and not kwargs.get("stream"):
            try:
                _log_completion_to_console(response)
            except Exception:
                log.debug("Console completion block failed", exc_info=True)
        return response

    AsyncCompletions.create = _patched_async_create

    sync_create = Completions.create

    def _patched_sync_create(self, *args, **kwargs):
        return sync_create(self, *args, **_apply_reasoning_policy(kwargs))

    Completions.create = _patched_sync_create

    _installed = True
    log.info("Reasoning control installed: thinking OFF by default, chat agent opts in.")