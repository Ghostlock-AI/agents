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

        # Add blank line before each tool (except first)
        print("", file=sys.stderr, flush=True)

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


def log_diff(old_content: str, new_content: str, filename: str, context_lines: int = 3, max_lines: int = 50, syntax_highlight: bool = True):
    """
    Log a colored unified diff showing changes made to a file.

    Args:
        old_content: Original file content
        new_content: Modified file content
        filename: Name of the file being edited
        context_lines: Number of context lines to show around changes
        max_lines: Maximum diff lines to show (default: 50)
        syntax_highlight: Enable syntax highlighting (default: True)

    This will output a git-style diff with:
    - Grey filename and line numbers
    - Red lines for deletions (-)
    - Green lines for additions (+)
    - Grey context lines (unchanged)
    - Syntax highlighting within code (if enabled)

    Example output:
        • Edited Dockerfile (+8 -1)
            9          curl \
            10    -    bash
            10    +    bash \
            11    +    jq \
            12    +    tar \
    """
    import difflib
    import re

    instance = ToolLogger()
    if not instance.enabled:
        return

    # Generate unified diff
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)

    diff = difflib.unified_diff(
        old_lines,
        new_lines,
        fromfile=filename,
        tofile=filename,
        n=context_lines,  # Context lines
        lineterm=''
    )

    diff_lines = list(diff)

    # Count changes
    additions = sum(1 for line in diff_lines if line.startswith('+') and not line.startswith('+++'))
    deletions = sum(1 for line in diff_lines if line.startswith('-') and not line.startswith('---'))

    # Check if diff is too large
    total_diff_lines = len([l for l in diff_lines if not l.startswith('---') and not l.startswith('+++') and not l.startswith('@@')])

    if total_diff_lines > max_lines:
        # Show summary for large diffs
        print(f"\033[2m  • Edited {filename} (+{additions} -{deletions})\033[0m", file=sys.stderr, flush=True)
        print(f"\033[2m    [Large diff: {total_diff_lines} lines changed, showing summary only]\033[0m", file=sys.stderr, flush=True)
        print(f"\033[2m    Changes: +{additions} additions, -{deletions} deletions\033[0m", file=sys.stderr, flush=True)
        print("", file=sys.stderr, flush=True)
        return

    # Print header
    print(f"\033[2m  • Edited {filename} (+{additions} -{deletions})\033[0m", file=sys.stderr, flush=True)

    # Get lexer for syntax highlighting
    lexer = None
    if syntax_highlight:
        lexer = _get_lexer_for_file(filename)

    # Process diff lines
    old_line_num = 0
    new_line_num = 0

    for line in diff_lines:
        if line.startswith('@@'):
            # Parse line numbers from hunk header: @@ -old_start,old_count +new_start,new_count @@
            match = re.search(r'@@ -(\d+),?\d* \+(\d+),?\d* @@', line)
            if match:
                old_line_num = int(match.group(1))
                new_line_num = int(match.group(2))
            continue
        elif line.startswith('---') or line.startswith('+++'):
            # Skip file headers
            continue

        # Print lines with color and line numbers
        line_content = line[1:].rstrip() if len(line) > 1 else ''

        # Apply syntax highlighting if enabled
        if lexer and line_content.strip():
            line_content = _highlight_code(line_content, lexer)

        if line.startswith('-'):
            # Deletion: show old line number, red
            print(f"\033[2m    {old_line_num:<4} \033[31m-   {line_content}\033[0m", file=sys.stderr, flush=True)
            old_line_num += 1
        elif line.startswith('+'):
            # Addition: show new line number, green
            print(f"\033[2m    {new_line_num:<4} \033[32m+   {line_content}\033[0m", file=sys.stderr, flush=True)
            new_line_num += 1
        else:
            # Context line: show new line number, grey
            print(f"\033[2m    {new_line_num:<4}     {line_content}\033[0m", file=sys.stderr, flush=True)
            old_line_num += 1
            new_line_num += 1

    print("", file=sys.stderr, flush=True)  # Blank line after diff


def _get_lexer_for_file(filename: str):
    """Get appropriate Pygments lexer for a file."""
    try:
        from pygments.lexers import get_lexer_for_filename
        from pygments.util import ClassNotFound

        try:
            return get_lexer_for_filename(filename, stripnl=False, stripall=False)
        except ClassNotFound:
            return None
    except ImportError:
        # Pygments not available
        return None


def _highlight_code(code: str, lexer) -> str:
    """Apply syntax highlighting to code using Pygments."""
    try:
        from pygments import highlight
        from pygments.formatters import TerminalFormatter

        # Use Terminal formatter with no background
        highlighted = highlight(code, lexer, TerminalFormatter())
        # Remove trailing newline that highlight() adds
        return highlighted.rstrip('\n')
    except Exception:
        # If highlighting fails, return original
        return code
