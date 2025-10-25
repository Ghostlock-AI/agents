"""
Data models for context management.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field
import hashlib


class FileState(BaseModel):
    """State tracking for a file in context."""

    filepath: str = Field(description="Absolute path to the file")
    content_hash: str = Field(description="SHA-256 hash of file content")
    version: int = Field(default=1, description="Version number for this file")
    last_updated: datetime = Field(default_factory=datetime.now, description="Last modification time")
    reference_count: int = Field(default=1, description="Number of times referenced")
    context_position: int = Field(default=0, description="Position in context buffer for stable ordering")
    token_count: Optional[int] = Field(default=None, description="Estimated token count")
    is_summarized: bool = Field(default=False, description="Whether file is currently summarized")
    summary: Optional[str] = Field(default=None, description="Summary of file content if compressed")

    @staticmethod
    def compute_hash(content: str) -> str:
        """Compute SHA-256 hash of content."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def update_content(self, new_content: str) -> None:
        """Update file with new content."""
        new_hash = self.compute_hash(new_content)
        if new_hash != self.content_hash:
            self.content_hash = new_hash
            self.version += 1
            self.last_updated = datetime.now()
            self.is_summarized = False
            self.summary = None


class FileReference(BaseModel):
    """Reference to a file with its content."""

    filepath: str
    content: str
    state: FileState

    @property
    def display_name(self) -> str:
        """Get display name for the file."""
        import os
        return os.path.basename(self.filepath)

    @property
    def is_compressed(self) -> bool:
        """Check if file is currently in compressed form."""
        return self.state.is_summarized


class ContextStats(BaseModel):
    """Statistics about current context usage."""

    total_files: int = Field(default=0, description="Total number of files in context")
    total_tokens: int = Field(default=0, description="Estimated total tokens")
    max_tokens: int = Field(default=100000, description="Maximum token budget")
    files_summarized: int = Field(default=0, description="Number of compressed files")
    utilization: float = Field(default=0.0, description="Context utilization (0.0-1.0)")

    def update_utilization(self) -> None:
        """Calculate current utilization percentage."""
        if self.max_tokens > 0:
            self.utilization = min(1.0, self.total_tokens / self.max_tokens)

    @property
    def is_near_limit(self) -> bool:
        """Check if context is nearing token limit (>70%)."""
        return self.utilization > 0.7

    @property
    def is_critical(self) -> bool:
        """Check if context is at critical level (>90%)."""
        return self.utilization > 0.9
