"""
Search tools: Grep (ripgrep), Glob (file finding)

These tools provide fast, structured searching with sensible defaults.
"""

import subprocess
from pathlib import Path
from typing import Optional
from langchain_core.tools import tool
from tool_logger import log_tool_start, log_tool_complete, log_tool_error


@tool
def grep_code(pattern: str, path: str = ".", file_pattern: Optional[str] = None, ignore_case: bool = False) -> str:
    """Search code using ripgrep (respects .gitignore automatically).

    Args:
        pattern: Regex pattern to search
        path: Directory to search (default: current directory)
        file_pattern: Optional glob to filter files (e.g., "*.py")
        ignore_case: Case-insensitive search

    Returns:
        Matching lines with file:line:content format

    Examples:
        grep_code("printf", ".", "*.c") - find printf in C files
        grep_code("TODO", "src/") - find TODOs in src/

    Why ripgrep:
    - 10x faster than grep
    - Auto-respects .gitignore (no node_modules spam)
    - Better regex support
    - Colored output for readability
    """
    log_tool_start("Grep", args_str=f"'{pattern}' in {path}")

    try:
        # Check if ripgrep is available
        check_rg = subprocess.run(["which", "rg"], capture_output=True)

        if check_rg.returncode != 0:
            # Fallback to grep if ripgrep not installed
            return _fallback_grep(pattern, path, file_pattern, ignore_case)

        cmd = ["rg", pattern, path, "--line-number", "--with-filename", "--no-heading"]

        if file_pattern:
            cmd.extend(["--glob", file_pattern])

        if ignore_case:
            cmd.append("--ignore-case")

        # Limit output to prevent overwhelming context
        cmd.extend(["--max-count", "50"])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

        if result.returncode == 0:
            matches = result.stdout.count('\n') if result.stdout else 0
            log_tool_complete(f"Found {matches} matches")
            return result.stdout if result.stdout else "No matches found"
        elif result.returncode == 1:
            log_tool_complete("No matches found")
            return "No matches found"
        else:
            log_tool_error(result.stderr)
            return f"Search error: {result.stderr}"

    except subprocess.TimeoutExpired:
        log_tool_error("Timeout")
        return "Search timed out (>10s)"
    except Exception as e:
        log_tool_error(str(e))
        return f"ERROR: {e}"


def _fallback_grep(pattern: str, path: str, file_pattern: Optional[str], ignore_case: bool) -> str:
    """Fallback to basic grep if ripgrep not available."""
    try:
        cmd = ["grep", "-rn", pattern, path]

        if ignore_case:
            cmd.insert(1, "-i")

        if file_pattern:
            cmd.extend(["--include", file_pattern])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

        if result.returncode == 0:
            lines = result.stdout.splitlines()[:50]  # Limit output
            return "\n".join(lines) if lines else "No matches found"
        else:
            return "No matches found"

    except subprocess.TimeoutExpired:
        return "Search timed out (>10s)"
    except Exception as e:
        return f"ERROR: {e}"


@tool
def find_files(pattern: str, path: str = ".") -> str:
    """Find files matching glob pattern.

    Args:
        pattern: Glob pattern (e.g., "**/*.py", "*.c")
        path: Directory to search

    Returns:
        List of matching file paths (sorted by modification time, newest first)

    Examples:
        find_files("**/*.c") - all C files recursively
        find_files("*.py", "src/") - Python files in src/

    Why this vs ls/find:
    - Respects .gitignore
    - Sorted by mtime (most recently edited first)
    - Simpler syntax than find
    """
    log_tool_start("FindFiles", args_str=f"'{pattern}' in {path}")

    try:
        base = Path(path).expanduser().resolve()

        if not base.exists():
            log_tool_error(f"Directory not found: {path}")
            return f"ERROR: Directory not found: {path}"

        if not base.is_dir():
            log_tool_error(f"Not a directory: {path}")
            return f"ERROR: Not a directory: {path}"

        # Use pathlib.glob with rglob for recursive
        if pattern.startswith("**/"):
            matches = base.rglob(pattern[3:])
        else:
            matches = base.glob(pattern)

        # Sort by modification time (newest first)
        files = sorted(matches, key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)

        # Filter out directories
        files = [f for f in files if f.is_file()]

        # Format output
        if not files:
            log_tool_complete("No files found")
            return "No files found"

        # Limit to 50 files
        file_list = files[:50]
        output = "\n".join(str(f.relative_to(base)) for f in file_list)

        if len(files) > 50:
            output += f"\n\n... and {len(files) - 50} more files (showing newest 50)"

        log_tool_complete(f"Found {len(files)} files")
        return output

    except Exception as e:
        log_tool_error(str(e))
        return f"ERROR: {e}"
