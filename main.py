import logging
import sys
from config import TOKEN

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

if __name__ == "__main__":
    if not TOKEN:
        logging.error("DISCORD_TOKEN not found in .env file.")
        sys.exit(1)

    from bot import bot

    bot.run(TOKEN)