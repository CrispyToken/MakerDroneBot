import asyncio
import logging
import os
import subprocess
import time

import aiohttp

log = logging.getLogger("rag-bot")


class InferenceEngine:
    """Manages the lifecycle of the llama.cpp `llama-server` binary."""

    def __init__(
        self,
        model_path: str,
        port: int = 8000,
        bin_path: str = "llama-server",
        api_key: str = "llamacpp",
        model_alias: str = "openai/local-model",
        n_gpu_layers: int = 999999,
        n_cpu_moe: int = 54,
        n_ctx: int = 32768,
        n_batch: int = 2048,
        n_ubatch: int = 512,
        n_threads: int = 14,
    ):
        self.model_path = model_path
        self.port = port
        self.bin_path = bin_path
        self.api_key = api_key
        self.model_alias = model_alias
        self.n_gpu_layers = n_gpu_layers
        self.n_cpu_moe = n_cpu_moe
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.n_ubatch = n_ubatch
        self.n_threads = n_threads
        self.process: subprocess.Popen | None = None

    async def start(self):
        if await self._is_running():
            log.info(f"llama-server already running on port {self.port}.")
            return

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"GGUF model not found at: {self.model_path}")
        if not os.path.exists(self.bin_path):
            raise FileNotFoundError(f"llama-server binary not found at: {self.bin_path}")

        log.info(
            f"Starting llama-server on port {self.port} "
            f"(n_gpu_layers={self.n_gpu_layers}, n_cpu_moe={self.n_cpu_moe}, n_ctx={self.n_ctx})..."
        )

        cmd = [
            self.bin_path,
            "--model", self.model_path,
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--api-key", self.api_key,
            "--alias", self.model_alias,
            "--no-webui",
            "--jinja",
            "--ctx-size", str(self.n_ctx),
            "--n-gpu-layers", str(self.n_gpu_layers),
            "--n-cpu-moe", str(self.n_cpu_moe),
            "--main-gpu", "0",
            "--split-mode", "layer",
            "--batch-size", str(self.n_batch),
            "--ubatch-size", str(self.n_ubatch),
            "--threads", str(self.n_threads),
            "--parallel", "1",
            "--cache-type-k", "q8_0",
            "--cache-type-v", "q8_0",
            "--flash-attn", "on",
            "--kv-offload",
            "--kv-unified",
            "--no-direct-io",
            "--mlock",
        ]

        self.process = subprocess.Popen(cmd, stdout=None, stderr=None)

        # 43GB mmap+mlock load takes a while
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