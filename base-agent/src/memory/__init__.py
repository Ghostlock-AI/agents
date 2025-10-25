"""
Memory Module

Handles long-term memory with vector storage for semantic retrieval.
"""

from .vector_store import VectorStore, SearchResult

__all__ = [
    "VectorStore",
    "SearchResult",
]
