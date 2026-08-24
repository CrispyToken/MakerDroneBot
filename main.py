import sys
import os
import asyncio
import logging
from urllib.parse import urlparse
from dotenv import load_dotenv
from config import TOKEN

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("rag-bot")


async def main():
    if not TOKEN:
        log.error("DISCORD_TOKEN not found in .env file.")
        sys.exit(1)

    # Install reasoning control before any LLM request can be made.
    from core.llm_reasoning import install_reasoning_patch
    install_reasoning_patch()

    # Determine if we need to manage the inference server lifecycle
    server_manager = os.getenv("LLM_SERVER_MANAGER", "external").lower()
    engine_manager = None

    if server_manager == "llamacpp":
        from services.inference_engine import InferenceEngine

        endpoint = os.getenv("LLM_ENDPOINT", "http://127.0.0.1:8000/v1")
        parsed_url = urlparse(endpoint)
        port = parsed_url.port or 8000

        engine_manager = InferenceEngine(
            model_path=os.getenv("LLAMA_MODEL_PATH", ""),
            port=port,
            bin_path=os.getenv("LLAMA_SERVER_BIN", "llama-server"),
            api_key=os.getenv("LLM_API_KEY", "llamacpp"),
            model_alias=os.getenv("LLM_MODEL", "openai/local-model"),
            n_gpu_layers=int(os.getenv("LLAMA_N_GPU_LAYERS", "999999")),
            n_cpu_moe=int(os.getenv("LLAMA_N_CPU_MOE", "54")),
            n_ctx=int(os.getenv("LLAMA_N_CTX", "32768")),
            n_batch=int(os.getenv("LLAMA_N_BATCH", "2048")),
            n_ubatch=int(os.getenv("LLAMA_N_UBATCH", "512")),
            n_threads=int(os.getenv("LLAMA_THREADS", "14")),
        )
        await engine_manager.start()

    elif server_manager == "freetoken":
        # Keeping this block intact in case you ever want to switch back to FreeToken
        from services.inference_engine import InferenceEngine
        endpoint = os.getenv("LLM_ENDPOINT", "http://127.0.0.1:1919/v1")
        parsed_url = urlparse(endpoint)
        port = parsed_url.port or 1919

        raw_model = os.getenv("LLM_MODEL", "nvidia/Qwen3.6-35B-A3B-NVFP4")
        model_path = raw_model.split("/", 1)[-1] if raw_model.startswith("openai/") else raw_model

        kv_tokens = int(os.getenv("MODEL_KV_TOKENS", "49152"))
        moe_cache_size = int(os.getenv("MODEL_MOE_CACHE_SIZE", "1024"))
        max_running_requests = int(os.getenv("MODEL_MAX_RUNNING_REQUESTS", "2"))

        engine_manager = InferenceEngine(
            model_path=model_path,
            port=port,
            kv_tokens=kv_tokens,
            moe_cache_size=moe_cache_size,
            max_running_requests=max_running_requests,
        )
        await engine_manager.start()
    else:
        log.info(
            f"LLM_SERVER_MANAGER is '{server_manager}'. Assuming external server at {os.getenv('LLM_ENDPOINT')} is already running.")

    try:
        from bot import bot
        await bot.start(TOKEN)
    finally:
        if engine_manager:
            await engine_manager.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Bot shutting down.")