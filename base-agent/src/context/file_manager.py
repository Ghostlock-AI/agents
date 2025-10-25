"""
File Context Manager

Manages file references in context with:
- Content-addressable deduplication
- Version tracking
- Token budget management
- Smart compression
"""

import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import tiktoken

from .models import FileState, FileReference, ContextStats


class FileContextManager:
    """Manages file context with deduplication and token tracking."""

    def __init__(
        self,
        max_tokens: int = 100000,
        compression_threshold: float = 0.7,
        encoding_name: str = "cl100k_base"
    ):
        """
        Initialize the file context manager.

        Args:
            max_tokens: Maximum token budget for context
            compression_threshold: Utilization threshold to trigger compression (0.0-1.0)
            encoding_name: Tiktoken encoding to use for token counting
        """
        self.max_tokens = max_tokens
        self.compression_threshold = compression_threshold

        # File registry: filepath -> FileState
        self._registry: Dict[str, FileState] = {}

        # Context buffer: ordered list of file paths
        self._context_buffer: List[str] = []

        # Token encoder
        try:
            self._encoder = tiktoken.get_encoding(encoding_name)
        except Exception:
            # Fallback to approximate counting
            self._encoder = None

        # Position counter for stable ordering
        self._position_counter = 0

    def add_file(self, filepath: str, content: Optional[str] = None) -> Tuple[bool, str]:
        """
        Add a file to context (or update if it already exists).

        Args:
            filepath: Path to the file
            content: File content (if None, will read from disk)

        Returns:
            Tuple of (success, message)
        """
        # Resolve absolute path
        abs_path = os.path.abspath(filepath)

        # Read content if not provided
        if content is None:
            try:
                with open(abs_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
            except Exception as e:
                return False, f"Error reading file: {e}"

        # Check if file already exists
        if abs_path in self._registry:
            # Update existing file
            old_state = self._registry[abs_path]
            old_hash = old_state.content_hash
            new_hash = FileState.compute_hash(content)

            if old_hash == new_hash:
                # Content unchanged, just increment reference count
                old_state.reference_count += 1
                old_state.last_updated = datetime.now()
                return True, f"File already in context (ref count: {old_state.reference_count})"
            else:
                # Content changed, update
                old_state.update_content(content)
                token_count = self._count_tokens(content)
                old_state.token_count = token_count
                return True, f"File updated to version {old_state.version}"
        else:
            # New file
            content_hash = FileState.compute_hash(content)
            token_count = self._count_tokens(content)

            state = FileState(
                filepath=abs_path,
                content_hash=content_hash,
                version=1,
                last_updated=datetime.now(),
                reference_count=1,
                context_position=self._position_counter,
                token_count=token_count,
                is_summarized=False
            )

            self._registry[abs_path] = state
            self._context_buffer.append(abs_path)
            self._position_counter += 1

            # Check if we need compression
            stats = self.get_stats()
            if stats.is_near_limit:
                self._compress_old_files()

            return True, f"File added to context ({token_count} tokens)"

    def remove_file(self, filepath: str) -> Tuple[bool, str]:
        """
        Remove a file from context.

        Args:
            filepath: Path to the file

        Returns:
            Tuple of (success, message)
        """
        abs_path = os.path.abspath(filepath)

        if abs_path not in self._registry:
            return False, "File not in context"

        # Remove from registry and buffer
        del self._registry[abs_path]
        self._context_buffer.remove(abs_path)

        return True, "File removed from context"

    def get_file(self, filepath: str) -> Optional[FileReference]:
        """
        Get a file reference from context.

        Args:
            filepath: Path to the file

        Returns:
            FileReference if found, None otherwise
        """
        abs_path = os.path.abspath(filepath)

        if abs_path not in self._registry:
            return None

        state = self._registry[abs_path]

        # Read content (either full or summary)
        if state.is_summarized and state.summary:
            content = state.summary
        else:
            try:
                with open(abs_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
            except Exception:
                content = f"[Error reading file: {abs_path}]"

        return FileReference(
            filepath=abs_path,
            content=content,
            state=state
        )

    def list_files(self) -> List[FileState]:
        """
        List all files currently in context.

        Returns:
            List of FileState objects ordered by context position
        """
        # Return in context buffer order
        return [
            self._registry[path]
            for path in self._context_buffer
            if path in self._registry
        ]

    def get_current_context(self) -> List[FileReference]:
        """
        Get all file references in context (deduplicated, ordered).

        Returns:
            List of FileReference objects
        """
        references = []

        for path in self._context_buffer:
            if path in self._registry:
                ref = self.get_file(path)
                if ref:
                    references.append(ref)

        return references

    def get_stats(self) -> ContextStats:
        """
        Get current context statistics.

        Returns:
            ContextStats object with current metrics
        """
        total_files = len(self._registry)
        files_summarized = sum(1 for state in self._registry.values() if state.is_summarized)
        total_tokens = sum(
            state.token_count or 0
            for state in self._registry.values()
        )

        stats = ContextStats(
            total_files=total_files,
            total_tokens=total_tokens,
            max_tokens=self.max_tokens,
            files_summarized=files_summarized
        )
        stats.update_utilization()

        return stats

    def clear(self) -> None:
        """Clear all files from context."""
        self._registry.clear()
        self._context_buffer.clear()
        self._position_counter = 0

    def _count_tokens(self, text: str) -> int:
        """
        Count tokens in text.

        Args:
            text: Text to count

        Returns:
            Estimated token count
        """
        if self._encoder:
            try:
                return len(self._encoder.encode(text))
            except Exception:
                pass

        # Fallback: rough approximation (4 chars per token)
        return len(text) // 4

    def _compress_old_files(self, target_utilization: float = 0.5) -> None:
        """
        Compress old files to reach target utilization.

        This summarizes files that haven't been updated recently,
        keeping only summaries in context.

        Args:
            target_utilization: Target utilization after compression
        """
        # Sort files by last_updated (oldest first)
        files_by_age = sorted(
            self._registry.items(),
            key=lambda x: x[1].last_updated
        )

        current_stats = self.get_stats()
        tokens_to_free = int(
            (current_stats.utilization - target_utilization) * self.max_tokens
        )

        tokens_freed = 0

        for filepath, state in files_by_age:
            if tokens_freed >= tokens_to_free:
                break

            if not state.is_summarized:
                # Summarize this file (use LLM if available)
                use_llm = hasattr(self, '_llm') and self._llm is not None
                llm = self._llm if use_llm else None
                summary = self._summarize_file(filepath, use_llm=use_llm, llm=llm)
                if summary:
                    original_tokens = state.token_count or 0
                    summary_tokens = self._count_tokens(summary)

                    state.is_summarized = True
                    state.summary = summary
                    state.token_count = summary_tokens

                    tokens_freed += (original_tokens - summary_tokens)

    def _summarize_file(self, filepath: str, use_llm: bool = False, llm=None) -> Optional[str]:
        """
        Create a summary of a file.

        Args:
            filepath: Path to file
            use_llm: Whether to use LLM for intelligent summarization
            llm: LLM instance to use for summarization (optional)

        Returns:
            Summary text
        """
        try:
            with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()

            lines = content.split('\n')

            # For small files, no summarization needed
            if len(lines) <= 20:
                return content

            # LLM-based summarization (if available and enabled)
            if use_llm and llm:
                try:
                    import os
                    filename = os.path.basename(filepath)

                    summary_prompt = f"""Summarize the following file concisely. Focus on:
1. Purpose and main functionality
2. Key components/functions/classes
3. Important dependencies or imports
4. Critical logic or algorithms

File: {filename}

Content:
{content[:4000]}  # Limit to first 4000 chars to avoid token limits

Provide a concise summary (max 200 words):"""

                    response = llm.invoke([{"role": "user", "content": summary_prompt}])
                    llm_summary = response.content if hasattr(response, 'content') else str(response)

                    # Create enhanced summary with structure
                    summary_parts = [
                        f"=== SUMMARY: {filename} ({len(lines)} lines) ===",
                        llm_summary.strip(),
                        "",
                        "=== FIRST 5 LINES ===",
                        '\n'.join(lines[:5]),
                        "",
                        "=== LAST 5 LINES ===",
                        '\n'.join(lines[-5:]),
                    ]

                    return '\n'.join(summary_parts)

                except Exception as e:
                    # Fall back to simple truncation if LLM fails
                    print(f"Warning: LLM summarization failed ({e}), using truncation")

            # Simple truncation strategy (fallback)
            summary_lines = (
                [f"=== FILE: {os.path.basename(filepath)} ({len(lines)} lines) ===", ""] +
                lines[:10] +
                ["", f"... [{len(lines) - 20} lines omitted] ...", ""] +
                lines[-10:]
            )

            return '\n'.join(summary_lines)

        except Exception:
            return f"[Error summarizing {filepath}]"

    def expand_file(self, filepath: str) -> Tuple[bool, str]:
        """
        Expand a summarized file to show full content.

        Args:
            filepath: Path to file

        Returns:
            Tuple of (success, message)
        """
        abs_path = os.path.abspath(filepath)

        if abs_path not in self._registry:
            return False, "File not in context"

        state = self._registry[abs_path]

        if not state.is_summarized:
            return False, "File is not summarized"

        # Read full content
        try:
            with open(abs_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()

            # Update state
            state.is_summarized = False
            state.summary = None
            state.token_count = self._count_tokens(content)

            return True, "File expanded to full content"

        except Exception as e:
            return False, f"Error expanding file: {e}"

    def format_for_context(self, include_summarized: bool = True) -> str:
        """
        Format all files for inclusion in LLM context.

        Args:
            include_summarized: Whether to include summarized files

        Returns:
            Formatted string with all file contents
        """
        parts = []
        references = self.get_current_context()

        for ref in references:
            if not include_summarized and ref.is_compressed:
                continue

            marker = " (SUMMARY)" if ref.is_compressed else ""
            parts.append(f"--- FILE: {ref.filepath}{marker} ---")
            parts.append(ref.content)
            parts.append("")

        return "\n".join(parts)
