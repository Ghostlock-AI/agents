"""
Persistent Memory System - ChromaDB-backed long-term memory for the agent.

This module provides tools for the agent to save and retrieve useful context
across sessions. The agent learns from successful solutions and user preferences,
building up a knowledge base of techniques and patterns.

Storage: ChromaDB (SQLite-based vector store)
Location: ./.memories/ in project root
Embeddings: OpenAI text-embedding-3-small
User Isolation: Supports multi-user via user_id filtering (currently single-user)
"""

import os
import logging
from pathlib import Path
from typing import List

# Disable ChromaDB telemetry before importing chromadb
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# Suppress ChromaDB telemetry error messages (version incompatibility in telemetry client)
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings


# ─── Determine Memory Storage Directory ────────────────────────────────────────
# Store memories in ./.memories/ at project root (base-agent/.memories/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # src/ -> base-agent/
PERSIST_DIR = PROJECT_ROOT / ".memories"
PERSIST_DIR.mkdir(parents=True, exist_ok=True)


# ─── Initialize ChromaDB Vector Store ──────────────────────────────────────────
"""
ChromaDB is like SQLite for vectors - lightweight, embedded, no setup required.
- Size: ~15MB for the library, 10s of MB when running
- Storage: DuckDB parquet files under the hood
- Auto-created on first use, reused on subsequent runs
- Perfect for local-first agent memory

NOTE: We use lazy initialization to ensure OpenAI API key is loaded before
creating the embeddings client. The vector store is created on first access.
"""
_persistent_memory_vector_store = None


def get_vector_store():
    """Get or create the persistent memory vector store (lazy initialization)."""
    global _persistent_memory_vector_store

    if _persistent_memory_vector_store is None:
        _persistent_memory_vector_store = Chroma(
            collection_name="agent_persisted_memories",
            embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
            persist_directory=str(PERSIST_DIR),
        )

    return _persistent_memory_vector_store


# ─── Helper: Extract User ID ───────────────────────────────────────────────────
def _get_user_id(config: RunnableConfig) -> str:
    """
    Extract user_id from LangGraph config for multi-user isolation.

    NOTE: Currently this agent is single-user (CLI-based), so user_id defaults to "1".
    In the future, if we support multiple users, we can:
    - Add CLI commands to switch users
    - Use environment variables or config files
    - Implement user authentication

    For now, all memories are stored under user_id="1".
    """
    user_id = config.get("configurable", {}).get("user_id", "1")
    if user_id is None:
        user_id = "1"  # Default to single user
    return str(user_id)


# ─── Tool: Save Persistent Memory ──────────────────────────────────────────────
@tool
def save_persistent_memory(memory: str, config: RunnableConfig) -> str:
    """Save CONCRETE, STEP-BY-STEP procedures to long-term memory.

    CRITICAL: Save ACTIONABLE, REPRODUCIBLE instructions that you can follow later.

    REQUIRED FORMAT - Save as detailed procedures:
    "To [accomplish X]: 1. [exact action with specific details], 2. [next action], 3. [result]"

    WHAT TO SAVE (with EXACT details):

    ✓ TOOL SEQUENCES with exact parameters:
      Example: "To get weather: 1. ddgs_search('weather [city] site:weather.com'),
      2. web_fetch('https://weather.com/weather/today/l/[location]'),
      3. Extract temperature from 'Current Weather' section"

    ✓ EXACT URLs that work:
      Example: "Weather data source: https://weather.com/weather/today/l/[city-code]"

    ✓ EXACT commands/code that work:
      Example: "To profile Python: python -m cProfile -o output.prof script.py && snakeviz output.prof"

    ✓ USER PREFERENCES with specifics:
      Example: "User wants Python code with: inline comments every 2-3 lines, type hints, docstrings"

    ✓ FILE PATHS and patterns:
      Example: "User's projects in ~/dev/ organized as: ~/dev/[language]/[project-name]/"

    WHAT NOT TO SAVE:
    ✗ Vague statements: "Use reliable sources" (not actionable)
    ✗ Ephemeral facts: "Weather is 72°F" (changes)
    ✗ General advice: "Search for information" (not specific)

    QUALITY CHECK before saving:
    - Can you follow these steps 6 months from now without any context?
    - Are all URLs, commands, and parameters included?
    - Is it a PROCEDURE (steps) not a PREFERENCE (opinion)?

    FOCUS: Save EXACT reproduction steps, not summaries.

    Args:
        memory: The information to remember (be specific and actionable)
        config: Runtime config (contains user_id for isolation)

    Returns:
        Confirmation message

    Example:
        save_persistent_memory(
            "User prefers Python code examples to be verbose with detailed comments",
            config
        )
        save_persistent_memory(
            "When debugging async Python code, use asyncio.run(debug=True) and check event loop state",
            config
        )
    """
    user_id = _get_user_id(config)

    # Create document with user_id metadata for filtering
    document = Document(
        page_content=memory,
        metadata={"user_id": user_id},
    )

    # Add to vector store (lazy init)
    get_vector_store().add_documents([document])

    return f"Memory saved: {memory}"


# ─── Tool: Search Persistent Memories ──────────────────────────────────────────
@tool
def search_persistent_memories(query: str, config: RunnableConfig) -> List[str]:
    """Search your memory for step-by-step procedures and relevant context.

    USE THIS TOOL PROACTIVELY - Check your memory BEFORE doing complex tasks!

    WHEN TO USE (be proactive):
    - BEFORE searching/fetching: "Have I done this before?"
    - BEFORE writing code: "Do I know the user's style preferences?"
    - BEFORE starting a task: "Is there a saved procedure for this?"
    - When user asks how to do something: Check if you have the exact steps saved
    - When user mentions a topic: Check if you have relevant context

    SEARCH EXAMPLES:
    - User asks "What's the weather?": search_persistent_memories("get weather")
    - User asks for Python code: search_persistent_memories("Python code preferences")
    - User asks "How do I X?": search_persistent_memories("how to X")
    - Starting any complex task: search_persistent_memories("[task description]")

    The search is semantic (meaning-based) - use natural language.
    Results are automatically filtered to this user's memories only.

    TIP: If you find a saved procedure, FOLLOW IT EXACTLY - it's proven to work!

    Args:
        query: What you want to recall (be specific)
        config: Runtime config (contains user_id for isolation)

    Returns:
        List of relevant memories (top 3 most similar)

    Example:
        search_persistent_memories("How does user prefer code examples?", config)
        search_persistent_memories("Python debugging techniques", config)
    """
    user_id = _get_user_id(config)

    # Search with user_id filter (only this user's memories) - lazy init
    documents = get_vector_store().similarity_search(
        query,
        k=3,
        filter={"user_id": user_id}
    )

    # Extract and return content
    return [doc.page_content for doc in documents]
