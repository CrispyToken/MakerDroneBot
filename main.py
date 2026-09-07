import sys
import asyncio
import logging
from config import (
    TOKEN,
    MODELS_DIR,
    LLM_SERVER_MANAGER,
    LLAMA_PORT,
    LLAMA_SERVER_BIN,
    LLM_API_KEY,
    LLM_DEFAULT_MODEL_PATH,
    LLM_INGEST_MODEL_PATH,
)

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

    engine_manager = None
    if LLM_SERVER_MANAGER == "llamacpp":
        from services.inference_engine import InferenceEngine
        engine_manager = InferenceEngine(
            model_rel_path=LLM_DEFAULT_MODEL_PATH,
            ingest_model_rel_path=LLM_INGEST_MODEL_PATH,
            port=LLAMA_PORT,
            bin_path=LLAMA_SERVER_BIN,
            api_key=LLM_API_KEY,
            models_dir=MODELS_DIR,
        )
        await engine_manager.start()
    else:
        log.info(
            f"LLM_SERVER_MANAGER is '{LLM_SERVER_MANAGER}'. "
            "Assuming external inference server is already running."
        )

    try:
        from bot import bot
        # Expose the engine manager so commands (e.g. !ingest) can drive
        # model swaps. None when running against an external server.
        bot.engine_manager = engine_manager
        await bot.start(TOKEN)
    finally:
        from core.memory import shutdown_memory
        await shutdown_memory()
        if engine_manager:
            await engine_manager.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Bot shutting down.")