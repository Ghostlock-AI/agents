"""
Context Management Module

Handles file context tracking, deduplication, and semantic retrieval.
"""

from .file_manager import FileContextManager
from .models import FileState, FileReference, ContextStats

__all__ = [
    "FileContextManager",
    "FileState",
    "FileReference",
    "ContextStats",
]
