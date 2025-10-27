"""
File operation tools: Read, Write, Edit

These tools provide structured, predictable file operations with better error handling
than generic shell commands.
"""

from langchain_core.tools import tool
from pathlib import Path
from typing import Optional
from tool_logger import log_tool_start, log_tool_complete, log_tool_error


@tool
def read_file(file_path: str, start_line: Optional[int] = None, end_line: Optional[int] = None) -> str:
    """Read file contents, optionally specific line range.

    Args:
        file_path: Path to file (absolute or relative to cwd)
        start_line: Optional starting line number (1-indexed)
        end_line: Optional ending line number (inclusive)

    Returns:
        File contents or specified line range

    Examples:
        read_file("hello.c") - read entire file
        read_file("main.py", 10, 20) - read lines 10-20
    """
    # Log tool start
    if start_line or end_line:
        log_tool_start("Read", args_str=f"{file_path}, lines {start_line}-{end_line}")
    else:
        log_tool_start("Read", args_str=file_path)

    try:
        path = Path(file_path).expanduser().resolve()

        if not path.exists():
            log_tool_error(f"File not found: {file_path}")
            return f"ERROR: File not found: {file_path}"

        if not path.is_file():
            log_tool_error(f"Not a file: {file_path}")
            return f"ERROR: Not a file: {file_path}"

        content = path.read_text()
        total_lines = content.count('\n') + 1

        if start_line or end_line:
            lines = content.splitlines()
            start = (start_line - 1) if start_line else 0
            end = end_line if end_line else len(lines)

            if start < 0 or start >= len(lines):
                log_tool_error(f"Line range out of bounds")
                return f"ERROR: start_line {start_line} out of range (file has {len(lines)} lines)"

            result = "\n".join(lines[start:end])
            log_tool_complete(f"Read {len(lines[start:end])} lines (of {total_lines} total)")
            return result

        log_tool_complete(f"Read {total_lines} lines")
        return content
    except Exception as e:
        log_tool_error(str(e))
        return f"ERROR reading {file_path}: {e}"


@tool
def write_file(file_path: str, content: str) -> str:
    """Create or overwrite file with content.

    Args:
        file_path: Path to file
        content: Full file contents

    Returns:
        Success message with file path

    Example:
        write_file("hello.c", '#include <stdio.h>\\n...')
    """
    log_tool_start("Write", args_str=file_path)

    try:
        path = Path(file_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

        lines = content.count('\n') + 1
        log_tool_complete(f"Wrote {lines} lines to {file_path}")
        return f"Successfully wrote {len(content)} bytes ({lines} lines) to {file_path}"
    except Exception as e:
        log_tool_error(str(e))
        return f"ERROR writing {file_path}: {e}"


@tool
def edit_file(file_path: str, old_string: str, new_string: str) -> str:
    """Edit file using exact string replacement.

    Args:
        file_path: Path to file
        old_string: Exact string to find (must match exactly)
        new_string: Replacement string

    Returns:
        Success message or error if old_string not found

    Example:
        edit_file("hello.c", 'printf("Hello', 'printf("Hello, World!\\n"')

    Why this is better than line-based editing:
    - More precise (exact match required)
    - No off-by-one line number errors
    - Works across multiple lines
    - Clear error if match fails
    """
    log_tool_start("Edit", args_str=file_path)

    try:
        path = Path(file_path).expanduser().resolve()

        if not path.exists():
            log_tool_error(f"File not found: {file_path}")
            return f"ERROR: File not found: {file_path}"

        content = path.read_text()

        if old_string not in content:
            # Show context to help debug
            preview = content[:300] + "..." if len(content) > 300 else content
            log_tool_error("String not found in file")
            return f"ERROR: old_string not found in {file_path}.\n\nFile preview:\n{preview}\n\nSearched for:\n{old_string}"

        # Count occurrences
        count = content.count(old_string)

        # Replace only first occurrence to be safe
        new_content = content.replace(old_string, new_string, 1)
        path.write_text(new_content)

        if count > 1:
            log_tool_complete(f"Replaced 1 of {count} occurrences in {file_path}")
            return f"SUCCESS: Replaced 1 of {count} occurrences in {file_path}. (Use multiple edit_file calls for other occurrences)"
        else:
            log_tool_complete(f"Replaced in {file_path}")
            return f"SUCCESS: Replaced in {file_path}"

    except Exception as e:
        log_tool_error(str(e))
        return f"ERROR editing {file_path}: {e}"
