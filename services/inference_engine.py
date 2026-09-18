import asyncio
import logging
import os
import subprocess
import time
from pathlib import Path
import aiohttp

log = logging.getLogger("rag-bot")

# Known boolean flags in llama-server that do NOT take a value argument.
# If your preset uses a boolean switch not listed here, add it to this set.
BOOL_FLAGS = {
    "mlock", "mmap", "no-mmap",
    "kv-offload", "no-kv-offload",
    "kv-unified", "no-kv-unified",
    "mmproj-offload", "no-mmproj-offload",
    "cpu-moe",
    "direct-io", "no-direct-io",
    "jinja", "no-jinja",
    "webui", "no-webui",
    "perf", "no-perf",
    "context-shift", "no-context-shift",
    "cont-batching", "no-cont-batching",
    "cache-prompt", "no-cache-prompt",
    "escape", "no-escape",
    "log-prefix", "no-log-prefix",
    "log-timestamps", "no-log-timestamps",
    "warmup", "no-warmup",
    "metrics", "props", "slots", "no-slots",
    "cors-credentials", "no-cors-credentials",
    "agent", "no-agent",
    "embedding", "embeddings", "rerank", "reranking",
    "skip-chat-parsing", "no-skip-chat-parsing",
    "prefill-assistant", "no-prefill-assistant",
    "reasoning-preserve", "no-reasoning-preserve",
}


def parse_preset_to_cli_args(preset_path: Path, model_filename: str) -> list[str]:
    """
    Parses a llama.cpp INI preset file and converts it to CLI arguments.
    Reads from the [*] (global) section and the section matching the model filename.
    Supports sharded models by matching if the section name is a prefix of the filename stem.
    """
    cli_args = []
    current_section = None
    model_stem = Path(model_filename).stem

    exact_targets = {'*', model_stem.lower(), model_filename.lower()}

    log.info(f"Parsing preset {preset_path.name} for model '{model_filename}' (stem: '{model_stem}')")

    try:
        with open(preset_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith(';'):
                    continue
                if line.startswith('[') and line.endswith(']'):
                    current_section_raw = line[1:-1].strip()
                    current_section_lower = current_section_raw.lower()

                    if current_section_lower in exact_targets or model_stem.lower().startswith(current_section_lower):
                        log.info(f" -> Applying INI section: [{current_section_raw}]")
                        current_section = current_section_raw
                    else:
                        current_section = None
                    continue

                if current_section is None:
                    continue

                if '=' in line:
                    key, value = line.split('=', 1)
                    key = key.strip().lstrip('-')  # Normalize if they included dashes
                    value = value.strip()
                    val_lower = value.lower()

                    if key in BOOL_FLAGS:
                        if val_lower in ('1', 'true', 'on', 'yes'):
                            cli_args.append(f"--{key}")
                        elif val_lower in ('0', 'false', 'off', 'no'):
                            if not key.startswith("no-"):
                                cli_args.append(f"--no-{key}")
                    else:
                        cli_args.extend([f"--{key}", value])
    except Exception as e:
        log.exception("Failed to parse preset.ini: %s", e)

    if not cli_args:
        log.warning("No preset arguments were loaded. Check that your INI section name matches the model filename.")
    else:
        log.info(f"Successfully loaded {len(cli_args)} preset arguments.")

    return cli_args


class InferenceEngine:
    """Manages the lifecycle of the llama.cpp `llama-server` binary.

    Supports hot-swapping between a default model and an optional, smaller
    ingest model. The server always advertises a single static alias
    ("local-model") so the OpenAI-compatible clients never need to know
    which physical GGUF is currently loaded.
    """

    # Static alias advertised via --alias. Must match LLM_MODEL_ID in config.py.
    MODEL_ALIAS = "local-model"

    def __init__(
        self,
        model_rel_path: str,
        port: int,
        bin_path: str,
        api_key: str,
        models_dir: Path,
        ingest_model_rel_path: str = "",
    ):
        self.models_dir = models_dir
        self.port = port
        self.bin_path = bin_path
        self.api_key = api_key

        self.model_alias = self.MODEL_ALIAS

        self.default_model_rel_path = model_rel_path
        self.ingest_model_rel_path = ingest_model_rel_path

        # Currently selected/loaded model (starts as the default).
        self.model_rel_path = model_rel_path
        self.current_model_rel_path = model_rel_path

        self.process: subprocess.Popen | None = None
        self._resolve_paths()

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------
    def _resolve_paths(self) -> None:
        """Derive model_path / preset_path from the current model_rel_path."""
        self.model_path = (self.models_dir / self.model_rel_path).resolve()
        self.preset_path = self.model_path.parent / "preset.ini"

    def _set_model(self, model_rel_path: str) -> None:
        """Point the engine at a different models/<...> entry."""
        self.model_rel_path = model_rel_path
        self._resolve_paths()

    def _process_alive(self) -> bool:
        """True when this engine owns a running llama-server process."""
        return self.process is not None and self.process.poll() is None

    # ------------------------------------------------------------------
    # Model swapping
    # ------------------------------------------------------------------
    async def swap_model(self, model_rel_path: str) -> None:
        """Stop the current llama-server and relaunch with a different model.

        If the new model fails to start, the previously loaded model is
        restored before the exception is re-raised, so the bot is never left
        without a working backend.
        """
        if not model_rel_path:
            log.info(f"InferenceEngine: no target model supplied, keeping '{self.current_model_rel_path}'.")
            return
        if model_rel_path == self.current_model_rel_path and self._process_alive():
            log.info(f"InferenceEngine: '{model_rel_path}' is already loaded, skipping swap.")
            return
        if not self._process_alive():
            raise RuntimeError(
                f"InferenceEngine: cannot swap models because this engine does not own a running "
                f"llama-server process on port {self.port}. If an external server occupies the port, "
                "free it and restart the bot, or use LLM_SERVER_MANAGER=external."
            )
        previous = self.current_model_rel_path
        log.info(f"InferenceEngine: swapping '{previous}' -> '{model_rel_path}'.")
        await self.stop()
        self._set_model(model_rel_path)
        try:
            await self.start()
        except (Exception, asyncio.CancelledError):
            log.exception(f"InferenceEngine: '{model_rel_path}' failed to load; rolling back to '{previous}'.")
            await self.stop()
            self._set_model(previous)
            try:
                await self.start()
            except Exception:
                log.exception(
                    f"InferenceEngine: rollback to '{previous}' also failed; no model is loaded. "
                    "A bot restart is likely required."
                )
            raise
        log.info(f"InferenceEngine: now serving '{self.current_model_rel_path}'.")

    async def load_default_model(self) -> None:
        await self.swap_model(self.default_model_rel_path)

    async def load_ingest_model(self) -> None:
        if not self.ingest_model_rel_path:
            log.info(f"InferenceEngine: no ingest model configured, staying on '{self.current_model_rel_path}'.")
            return
        await self.swap_model(self.ingest_model_rel_path)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def start(self):
        if await self._is_running():
            raise RuntimeError(
                f"Port {self.port} is already serving an OpenAI-compatible API. "
                "Stop that process, or set LLM_SERVER_MANAGER=external if it is intentional."
            )
        if not os.path.exists(self.bin_path):
            raise FileNotFoundError(f"llama-server binary not found at: {self.bin_path}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"GGUF model not found at: {self.model_path}")
        if not os.path.exists(self.preset_path):
            raise FileNotFoundError(f"Preset INI not found at: {self.preset_path}")

        log.info(f"Parsing preset INI: {self.preset_path.name}")
        preset_args = parse_preset_to_cli_args(self.preset_path, self.model_path.name)

        cmd = [
            self.bin_path,
            "--model", str(self.model_path),
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--api-key", self.api_key,
            "--alias", self.model_alias,
            "--no-webui",
            "--jinja",
        ]
        # Append the dynamically generated preset arguments
        cmd.extend(preset_args)

        log.info(f"Starting llama-server on port {self.port} with {len(preset_args)} preset arguments.")
        self.process = subprocess.Popen(cmd, stdout=None, stderr=None)
        await self._wait_for_healthy(timeout=300)
        self.current_model_rel_path = self.model_rel_path

    async def stop(self):
        if self.process and self.process.poll() is None:
            log.info("Shutting down llama-server...")
            self.process.terminate()
            try:
                await asyncio.to_thread(self.process.wait, timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            log.info("llama-server stopped.")
        self.process = None
        # Ensure the port is actually released before any restart.
        await self._wait_for_stopped(timeout=30)

    # ------------------------------------------------------------------
    # Health checks
    # ------------------------------------------------------------------
    def _auth_headers(self):
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    async def _is_running(self) -> bool:
        url = f"http://127.0.0.1:{self.port}/v1/models"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=self._auth_headers()) as response:
                    return response.status == 200
        except Exception:
            return False

    async def _wait_for_stopped(self, timeout: int = 30):
        """Wait until the server no longer answers on its port."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not await self._is_running():
                return
            await asyncio.sleep(0.5)
        log.warning(f"llama-server still answering on port {self.port} after stop; restart may fail.")

    async def _wait_for_healthy(self, timeout: int = 300):
        base = f"http://127.0.0.1:{self.port}/v1"
        models_url = f"{base}/models"
        chat_url = f"{base}/chat/completions"
        deadline = time.monotonic() + timeout
        headers = self._auth_headers()

        payload = {
            "model": self.model_alias,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
            "stream": False,
        }
        request_timeout = aiohttp.ClientTimeout(total=60)

        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited prematurely with code {self.process.returncode}. Check console output."
                )
            try:
                async with aiohttp.ClientSession(timeout=request_timeout) as session:
                    async with session.get(models_url, headers=headers) as resp:
                        if resp.status == 200:
                            async with session.post(chat_url, json=payload, headers=headers) as chat_resp:
                                if chat_resp.status == 200:
                                    log.info("llama-server is fully ready.")
                                    return
            except Exception:
                pass
            await asyncio.sleep(2)

        raise RuntimeError(f"llama-server did not become healthy within {timeout}s.")