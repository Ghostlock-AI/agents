"""
Tool Logger - Real-time logging for tool execution.

Provides visual feedback when tools are called, similar to Claude Code's interface.
Logs are written to stderr so they don't interfere with tool return values.
"""

import sys
from typing import Optional
from datetime import datetime


class ToolLogger:
    """
    Singleton logger for tool execution events.

    Outputs to stderr with formatted indicators:
    ⏺ ToolName(args)
      ⎿  Result summary
    """

    _instance = None
    _enabled = True

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.enabled = True
        self.compact = True  # Compact format (no timestamps)

    @classmethod
    def log_start(cls, tool_name: str, description: str = "", args_str: str = ""):
        """
        Log when a tool starts executing.

        Args:
            tool_name: Name of the tool (e.g., "Read", "Bash", "Write")
            description: Human-readable description
            args_str: String representation of arguments

        Example:
            log_start("Read", args_str="src/tools.py")
            # Output: ⏺ Read(src/tools.py)
        """
        instance = cls()
        if not instance.enabled:
            return

        if args_str:
            print(f"\033[2m⏺ {tool_name}({args_str})\033[0m", file=sys.stderr, flush=True)
        else:
            print(f"\033[2m⏺ {tool_name}\033[0m", file=sys.stderr, flush=True)

    @classmethod
    def log_complete(cls, summary: str):
        """
        Log when a tool completes successfully.

        Args:
            summary: One-line summary of what happened

        Example:
            log_complete("Read 158 lines")
            # Output:   ⎿  Read 158 lines
        """
        instance = cls()
        if not instance.enabled:
            return

        print(f"\033[2m  ⎿  {summary}\033[0m", file=sys.stderr, flush=True)

    @classmethod
    def log_error(cls, error_msg: str):
        """
        Log when a tool encounters an error.

        Args:
            error_msg: Error description

        Example:
            log_error("File not found: src/missing.py")
            # Output:   ⎿  ERROR: File not found: src/missing.py
        """
        instance = cls()
        if not instance.enabled:
            return

        print(f"\033[31m  ⎿  ERROR: {error_msg}\033[0m", file=sys.stderr, flush=True)

    @classmethod
    def log_info(cls, info: str):
        """
        Log informational message (indented, dim).

        Args:
            info: Information to display
        """
        instance = cls()
        if not instance.enabled:
            return

        print(f"\033[2m     {info}\033[0m", file=sys.stderr, flush=True)

    @classmethod
    def enable(cls):
        """Enable tool logging."""
        instance = cls()
        instance.enabled = True

    @classmethod
    def disable(cls):
        """Disable tool logging."""
        instance = cls()
        instance.enabled = False

    @classmethod
    def is_enabled(cls) -> bool:
        """Check if logging is enabled."""
        instance = cls()
        return instance.enabled


# Convenience functions for direct import
def log_tool_start(tool_name: str, description: str = "", args_str: str = ""):
    """Log tool execution start."""
    ToolLogger.log_start(tool_name, description, args_str)


def log_tool_complete(summary: str):
    """Log tool execution completion."""
    ToolLogger.log_complete(summary)


def log_tool_error(error_msg: str):
    """Log tool execution error."""
    ToolLogger.log_error(error_msg)


def log_tool_info(info: str):
    """Log tool informational message."""
    ToolLogger.log_info(info)


def format_bytes(bytes_count: int) -> str:
    """
    Format byte count into human-readable string.

    Examples:
        128 → "128B"
        1500 → "1.5KB"
        1048576 → "1.0MB"
    """
    if bytes_count < 1024:
        return f"{bytes_count}B"
    elif bytes_count < 1024 * 1024:
        return f"{bytes_count / 1024:.1f}KB"
    elif bytes_count < 1024 * 1024 * 1024:
        return f"{bytes_count / (1024 * 1024):.1f}MB"
    else:
        return f"{bytes_count / (1024 * 1024 * 1024):.1f}GB"


def truncate_for_display(text: str, max_length: int = 50) -> str:
    """
    Truncate text for display in logs.

    Args:
        text: Text to truncate
        max_length: Maximum length before truncation

    Returns:
        Truncated text with ellipsis if needed
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."
