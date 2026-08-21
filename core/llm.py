import asyncio
import aiohttp
import time
import logging
from config import (
    LMSTUDIO_BASE_URL, LMSTUDIO_API_KEY, FAST_MODEL_ID, DEEP_MODEL_ID,
    MODEL_SWITCH_MODE, FAST_MODEL_LOAD_COMMAND, DEEP_MODEL_LOAD_COMMAND,
    MODEL_SWITCH_TIMEOUT, MODEL_SWITCH_SETTLE_SECONDS
)

log = logging.getLogger("rag-bot")
MODEL_LOCK = asyncio.Lock()

def get_auth_headers() -> dict:
    if LMSTUDIO_API_KEY: return {"Authorization": f"Bearer {LMSTUDIO_API_KEY}"}
    return {}

async def get_loaded_model() -> str | None:
    url = f"{LMSTUDIO_BASE_URL}/models"
    headers = get_auth_headers()
    try:
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get("data", [])
                    if not models: return None
                    loaded_ids = [m.get("id") for m in models if m.get("id")]
                    if FAST_MODEL_ID in loaded_ids: return FAST_MODEL_ID
                    if DEEP_MODEL_ID in loaded_ids: return DEEP_MODEL_ID
                    return loaded_ids[0] if loaded_ids else None
    except Exception as e:
        log.warning("Could not query LM Studio for loaded models: %s", e)
    return None

async def run_shell_command(command: str) -> None:
    log.info("Running model switch command: %s", command)
    process = await asyncio.create_subprocess_shell(command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=MODEL_SWITCH_TIMEOUT)
    except asyncio.TimeoutError:
        process.kill()
        raise RuntimeError(f"Model switch command timed out after {MODEL_SWITCH_TIMEOUT}s: {command}")
    if process.returncode != 0:
        error_text = stderr.decode("utf-8", errors="ignore").strip()
        raise RuntimeError(f"Model switch command failed with code {process.returncode}: {error_text}")
    output_text = stdout.decode("utf-8", errors="ignore").strip()
    if output_text: log.info("Model switch output: %s", output_text[:500])

async def wait_for_lmstudio(timeout: int = 30) -> None:
    deadline = time.monotonic() + timeout
    url = f"{LMSTUDIO_BASE_URL}/models"
    headers = get_auth_headers()
    while time.monotonic() < deadline:
        try:
            client_timeout = aiohttp.ClientTimeout(total=5)
            async with aiohttp.ClientSession(timeout=client_timeout) as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200: return
        except Exception:
            pass
        await asyncio.sleep(1)
    raise RuntimeError("LM Studio did not become reachable after model switching.")

async def ensure_model(model_kind: str) -> str:
    model_kind = model_kind.lower()
    if model_kind == "deep":
        target_model_id, load_command = DEEP_MODEL_ID, DEEP_MODEL_LOAD_COMMAND
    else:
        target_model_id, load_command = FAST_MODEL_ID, FAST_MODEL_LOAD_COMMAND

    if MODEL_SWITCH_MODE == "none": return target_model_id
    if MODEL_SWITCH_MODE != "command": raise RuntimeError(f"Unknown MODEL_SWITCH_MODE: {MODEL_SWITCH_MODE}")

    currently_loaded = await get_loaded_model()
    log.info("LM Studio currently has loaded: %s", currently_loaded or "None")
    if currently_loaded == target_model_id: return target_model_id

    if not load_command:
        log.warning("No load command configured for %s model. Assuming LM Studio will handle model selection.", model_kind)
        return target_model_id

    log.info("Target model %s not loaded. Running load command.", target_model_id)
    await run_shell_command(load_command)
    if MODEL_SWITCH_SETTLE_SECONDS > 0: await asyncio.sleep(MODEL_SWITCH_SETTLE_SECONDS)
    await wait_for_lmstudio(timeout=30)
    new_loaded = await get_loaded_model()
    log.info("After load command, LM Studio has: %s", new_loaded or "None")
    return target_model_id