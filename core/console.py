"""
LM Studio-style console output.

Complete structured blocks are printed AFTER each completion finishes.
Every block is assembled into a single string and written with one sys.stdout.write,
so llama-server's stats lines can never split a block mid-line.
"""
import sys
import logging

log = logging.getLogger("rag-bot")

_WIDTH = 78

_RESET = "\033[0m"
_DIM_CYAN = "\033[2;36m"   # thinking
_YELLOW = "\033[33m"       # tool calls
_BOLD = "\033[1m"          # labels


def _box(label: str, body: str, color: str) -> str:
    top = f"╭─ {label} "
    top += "─" * max(2, _WIDTH - len(top))
    lines = [color + top + _RESET]
    for line in body.strip().splitlines():
        lines.append(color + "│" + _RESET + " " + line)
    lines.append(color + "╰" + "─" * (_WIDTH - 1) + _RESET)
    return "\n".join(lines) + "\n"


def print_block(label: str, body: str, color: str = "") -> None:
    body = (body or "").strip()
    if not body:
        return
    sys.stdout.write(_box(label, body, color))
    sys.stdout.flush()


def print_user_line(text: str) -> None:
    preview = (text or "").replace("\n", " ")
    if len(preview) > 140:
        preview = preview[:140] + "…"
    sys.stdout.write(f"{_BOLD}[user]{_RESET} {preview}\n")
    sys.stdout.flush()


def print_completion(reasoning: str, content: str,
                     tool_calls: list | None = None,
                     source: str = "chat") -> None:
    """Print one finished completion as structured blocks."""
    if reasoning:
        print_block(f"{source} · thinking", reasoning, _DIM_CYAN)
    for name, args in (tool_calls or []):
        print_block(f"{source} · tool call", f"{name}({args})", _YELLOW)
    if content:
        print_block(f"{source} · reply", content)