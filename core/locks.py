"""
Global LLM lock.

One asyncio.Lock shared by every subsystem that occupies the local
inference engine: chat turns, monitoring evaluations, and document
ingestion. Whoever acquires it keeps it until fully done, then releases.
"""

import asyncio

llm_lock = asyncio.Lock()