"""
Directory tools: list, tree view

Provides directory exploration and navigation capabilities.
"""

from langchain_core.tools import tool
from pathlib import Path
from typing import Optional, List
from datetime import datetime
from tool_logger import log_tool_start, log_tool_complete, log_tool_error, format_bytes


@tool
def list_directory(path: str = ".", pattern: Optional[str] = None, show_hidden: bool = False) -> str:
    """List directory contents with file sizes and modification times.

    Args:
        path: Directory to list (default: current directory)
        pattern: Optional glob pattern to filter (e.g., "*.py", "test_*")
        show_hidden: Include hidden files/directories (default: False)

    Returns:
        Formatted directory listing with metadata

    Examples:
        list_directory() - list current directory
        list_directory("src/") - list src directory
        list_directory(".", "*.py") - list Python files only
        list_directory(".", show_hidden=True) - include hidden files

    Why use this vs bash 'ls':
    - Structured, consistent output format
    - File size in human-readable format (KB, MB)
    - Relative timestamps (2m ago, 1h ago)
    - Automatic sorting (directories first, then by name)
    """
    args_parts = [path]
    if pattern:
        args_parts.append(f"pattern={pattern}")
    log_tool_start("List", args_str=", ".join(args_parts))

    try:
        base = Path(path).expanduser().resolve()

        if not base.exists():
            log_tool_error(f"Directory not found: {path}")
            return f"ERROR: Directory not found: {path}"

        if not base.is_dir():
            log_tool_error(f"Not a directory: {path}")
            return f"ERROR: Not a directory: {path}"

        # Get entries
        if pattern:
            entries = list(base.glob(pattern))
        else:
            entries = list(base.iterdir())

        # Filter hidden files
        if not show_hidden:
            entries = [e for e in entries if not e.name.startswith('.')]

        if not entries:
            log_tool_complete("Empty directory")
            return f"Directory is empty: {path}"

        # Sort: directories first, then files, both alphabetically
        dirs = sorted([e for e in entries if e.is_dir()], key=lambda x: x.name)
        files = sorted([e for e in entries if e.is_file()], key=lambda x: x.name)

        # Format output
        lines = []
        lines.append(f"Directory: {base}\n")

        # List directories
        if dirs:
            lines.append("Directories:")
            for d in dirs:
                mtime = datetime.fromtimestamp(d.stat().st_mtime)
                mtime_str = _format_time_ago(mtime)
                lines.append(f"  📁 {d.name}/ (modified {mtime_str})")
            lines.append("")

        # List files
        if files:
            lines.append("Files:")
            for f in files:
                stat = f.stat()
                size_str = format_bytes(stat.st_size)
                mtime = datetime.fromtimestamp(stat.st_mtime)
                mtime_str = _format_time_ago(mtime)
                lines.append(f"  📄 {f.name} ({size_str}, modified {mtime_str})")
            lines.append("")

        # Summary
        total_files = len(files)
        total_dirs = len(dirs)
        lines.append(f"Total: {total_files} files, {total_dirs} directories")

        result = "\n".join(lines)
        log_tool_complete(f"{total_files} files, {total_dirs} dirs")

        return result

    except Exception as e:
        log_tool_error(str(e))
        return f"ERROR: {e}"


@tool
def tree_view(path: str = ".", max_depth: int = 3, show_hidden: bool = False) -> str:
    """Show directory tree structure (ASCII visualization).

    Args:
        path: Root directory to show tree for
        max_depth: Maximum depth to traverse (default: 3)
        show_hidden: Include hidden files/directories (default: False)

    Returns:
        ASCII tree visualization of directory structure

    Examples:
        tree_view() - show tree of current directory
        tree_view("src/", 2) - show src/ tree, max 2 levels deep
        tree_view(".", 5, True) - show full tree including hidden files

    Output format:
        .
        ├── src/
        │   ├── main.py
        │   ├── utils.py
        │   └── tools/
        │       ├── file_ops.py
        │       └── search.py
        └── tests/
            └── test_main.py
    """
    args_parts = [path, f"depth={max_depth}"]
    log_tool_start("Tree", args_str=", ".join(args_parts))

    try:
        base = Path(path).expanduser().resolve()

        if not base.exists():
            log_tool_error(f"Directory not found: {path}")
            return f"ERROR: Directory not found: {path}"

        if not base.is_dir():
            log_tool_error(f"Not a directory: {path}")
            return f"ERROR: Not a directory: {path}"

        lines = [str(base)]
        _build_tree(base, lines, "", max_depth, 0, show_hidden)

        result = "\n".join(lines)

        # Count items
        item_count = len(lines) - 1  # Subtract root
        log_tool_complete(f"{item_count} items")

        return result

    except Exception as e:
        log_tool_error(str(e))
        return f"ERROR: {e}"


def _build_tree(directory: Path, lines: List[str], prefix: str, max_depth: int, current_depth: int, show_hidden: bool):
    """Recursively build tree structure."""
    if current_depth >= max_depth:
        return

    try:
        # Get entries
        entries = list(directory.iterdir())

        # Filter hidden
        if not show_hidden:
            entries = [e for e in entries if not e.name.startswith('.')]

        # Sort: directories first
        entries = sorted(entries, key=lambda x: (not x.is_dir(), x.name))

        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1

            # Tree characters
            if is_last:
                connector = "└── "
                extension = "    "
            else:
                connector = "├── "
                extension = "│   "

            # Add entry
            if entry.is_dir():
                lines.append(f"{prefix}{connector}{entry.name}/")
                # Recurse
                _build_tree(entry, lines, prefix + extension, max_depth, current_depth + 1, show_hidden)
            else:
                lines.append(f"{prefix}{connector}{entry.name}")

    except PermissionError:
        lines.append(f"{prefix}[Permission Denied]")


def _format_time_ago(dt: datetime) -> str:
    """Format datetime as relative time (e.g., '2m ago', '3h ago')."""
    now = datetime.now()
    diff = now - dt

    seconds = diff.total_seconds()

    if seconds < 60:
        return "just now"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        return f"{minutes}m ago"
    elif seconds < 86400:
        hours = int(seconds / 3600)
        return f"{hours}h ago"
    elif seconds < 604800:
        days = int(seconds / 86400)
        return f"{days}d ago"
    else:
        weeks = int(seconds / 604800)
        return f"{weeks}w ago"
