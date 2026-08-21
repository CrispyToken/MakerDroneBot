import aiohttp
import logging
from config import SEARXNG_URL

log = logging.getLogger("rag-bot")

async def execute_web_search(query: str) -> str:
    url = f"{SEARXNG_URL}/search"
    params = {"q": query, "format": "json", "categories": "general"}
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"}
    try:
        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url, params=params, headers=headers) as resp:
                if resp.status != 200:
                    return f"Web search failed due to a server error (Status {resp.status})."
                data = await resp.json()
                results = data.get("results", [])[:5]
                if not results: return "No relevant search results found for this query."
                formatted_results = [
                    f"Title: {r.get('title', 'No title')}\nURL: {r.get('url', '')}\nSnippet: {r.get('content', 'No snippet available')}"
                    for r in results]
                return "\n\n".join(formatted_results)
    except Exception as e:
        log.exception("SearXNG request failed")
        return f"Web search failed: {str(e)}"