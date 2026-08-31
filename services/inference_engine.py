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
    """
    cli_args = []
    current_section = None
    model_stem = Path(model_filename).stem

    # We want to collect args from [*] and the specific model section
    target_sections = {'*', model_stem, model_filename}

    try:
        with open(preset_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith(';'):
                    continue
                if line.startswith('[') and line.endswith(']'):
                    current_section = line[1:-1].strip()
                    continue

                if current_section not in target_sections:
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

    return cli_args


class InferenceEngine:
    """Manages the lifecycle of the llama.cpp `llama-server` binary."""

    def __init__(
            self,
            model_rel_path: str,
            port: int,
            bin_path: str,
            api_key: str,
            models_dir: Path,
    ):
        self.models_dir = models_dir
        self.model_rel_path = model_rel_path

        # Resolve absolute paths based on the models directory
        self.model_path = (self.models_dir / self.model_rel_path).resolve()
        self.preset_path = self.model_path.parent / "preset.ini"

        self.port = port
        self.bin_path = bin_path
        self.api_key = api_key
        self.model_alias = self.model_rel_path

        self.process: subprocess.Popen | None = None

    async def start(self):
        if await self._is_running():
            log.info(f"llama-server already running on port {self.port}.")
            return

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

    async def stop(self):
        if self.process and self.process.poll() is None:
            log.info("Shutting down llama-server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            log.info("llama-server stopped.")

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