"""
Patch tool: Apply unified diffs

Enables applying git-style patches to files for structured edits.
"""

from langchain_core.tools import tool
from pathlib import Path
from typing import Optional
import subprocess
import tempfile
from tool_logger import log_tool_start, log_tool_complete, log_tool_error, log_tool_info


@tool
def apply_patch(patch_content: str, check_only: bool = False, reverse: bool = False) -> str:
    """Apply unified diff patch to files (git-style patches).

    Args:
        patch_content: Unified diff content (output from 'git diff' or 'diff -u')
        check_only: Only verify if patch applies cleanly (dry run, default: False)
        reverse: Apply patch in reverse (undo a patch, default: False)

    Returns:
        Success message with affected files or error with line numbers

    Examples:
        apply_patch(diff_text) - apply patch to files
        apply_patch(diff_text, check_only=True) - verify patch applies cleanly
        apply_patch(diff_text, reverse=True) - undo a previously applied patch

    Patch format (unified diff):
        --- a/file.py
        +++ b/file.py
        @@ -10,3 +10,4 @@
         def hello():
        -    print("hello")
        +    print("hello, world")
        +    return True

    Why use patches:
    - Apply multi-file changes atomically
    - Verify changes before applying (check_only)
    - Undo changes (reverse)
    - Standard git workflow compatibility
    """
    args_parts = []
    if check_only:
        args_parts.append("check_only=True")
    if reverse:
        args_parts.append("reverse=True")

    log_tool_start("ApplyPatch", args_str=", ".join(args_parts) if args_parts else "")

    try:
        # Validate patch content
        if not patch_content.strip():
            log_tool_error("Empty patch content")
            return "ERROR: patch_content cannot be empty"

        # Check if patch looks like unified diff
        if not ("---" in patch_content and "+++" in patch_content):
            log_tool_error("Not a valid unified diff format")
            return "ERROR: patch_content does not appear to be a unified diff (missing --- and +++ headers)"

        # Write patch to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.patch', delete=False) as f:
            patch_file = f.name
            f.write(patch_content)

        try:
            # Try using git apply first (better error messages)
            cmd = ["git", "apply"]

            if check_only:
                cmd.append("--check")

            if reverse:
                cmd.append("--reverse")

            # Verbose output for better error messages
            cmd.extend(["--verbose", patch_file])

            log_tool_info(f"Attempting: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=10
            )

            # Clean up temp file
            Path(patch_file).unlink()

            if result.returncode == 0:
                # Success
                if check_only:
                    summary = "Patch applies cleanly (dry run)"
                    log_tool_complete(summary)
                    return f"SUCCESS: {summary}\n\n{result.stdout if result.stdout else 'No output'}"
                else:
                    # Parse affected files from output
                    affected_files = _parse_affected_files(patch_content)

                    summary = f"Applied patch to {len(affected_files)} file(s)"
                    log_tool_complete(summary)

                    return f"""SUCCESS: Patch applied successfully

Affected files: {len(affected_files)}
{chr(10).join(f'  - {f}' for f in affected_files)}

{result.stdout if result.stdout else ''}
"""
            else:
                # Failed
                error_msg = result.stderr if result.stderr else result.stdout
                log_tool_error("Patch failed to apply")

                return f"""ERROR: Patch failed to apply

{error_msg}

This usually means:
  - The file content doesn't match the patch context
  - The patch was already applied
  - Line numbers have changed

Try:
  1. Verify the files haven't been modified since the patch was created
  2. Use check_only=True to see detailed errors
  3. Manually inspect the affected files
"""

        except subprocess.TimeoutExpired:
            Path(patch_file).unlink()
            log_tool_error("Patch operation timed out")
            return "ERROR: Patch operation timed out (>10s)"

        except FileNotFoundError:
            # git not available, try using patch command
            log_tool_info("git not found, trying 'patch' command")
            return _apply_with_patch_command(patch_content, check_only, reverse, patch_file)

    except Exception as e:
        log_tool_error(str(e))
        return f"ERROR: {e}"


def _apply_with_patch_command(patch_content: str, check_only: bool, reverse: bool, patch_file: str) -> str:
    """Fallback to using 'patch' command if git is not available."""
    try:
        cmd = ["patch"]

        if check_only:
            cmd.append("--dry-run")

        if reverse:
            cmd.append("--reverse")

        # Strip level (for 'a/file' and 'b/file' prefixes)
        cmd.extend(["-p1", "-i", patch_file])

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10
        )

        # Clean up
        Path(patch_file).unlink()

        if result.returncode == 0:
            affected_files = _parse_affected_files(patch_content)

            if check_only:
                summary = "Patch applies cleanly (dry run)"
            else:
                summary = f"Applied patch to {len(affected_files)} file(s)"

            log_tool_complete(summary)

            return f"""SUCCESS: {summary}

{result.stdout if result.stdout else 'Patch applied'}
"""
        else:
            log_tool_error("Patch command failed")
            return f"ERROR: Patch failed\n{result.stderr if result.stderr else result.stdout}"

    except FileNotFoundError:
        log_tool_error("Neither 'git' nor 'patch' command available")
        return "ERROR: Neither 'git apply' nor 'patch' command is available on this system"
    except subprocess.TimeoutExpired:
        log_tool_error("Timeout")
        return "ERROR: Patch operation timed out"
    except Exception as e:
        log_tool_error(str(e))
        return f"ERROR: {e}"


def _parse_affected_files(patch_content: str) -> list:
    """Extract list of affected files from patch content."""
    files = []
    for line in patch_content.splitlines():
        # Look for +++ lines (new file version)
        if line.startswith("+++"):
            # Extract filename (skip 'b/' prefix if present)
            parts = line.split()
            if len(parts) >= 2:
                filename = parts[1]
                if filename.startswith("b/"):
                    filename = filename[2:]
                if filename not in files:
                    files.append(filename)
    return files


@tool
def diff_files(file1: str, file2: Optional[str] = None, unified: bool = True, context_lines: int = 3) -> str:
    """Generate unified diff between files or file versions.

    Args:
        file1: First file path (or current file for git diff)
        file2: Second file path (or None for git diff against HEAD)
        unified: Use unified diff format (default: True)
        context_lines: Lines of context around changes (default: 3)

    Returns:
        Unified diff output or error message

    Examples:
        diff_files("old.py", "new.py") - diff two files
        diff_files("src/main.py") - git diff for file (if in git repo)

    Why use this:
    - Preview changes before committing
    - Generate patches for apply_patch tool
    - Compare file versions
    - Standard diff format for all tools
    """
    if file2:
        log_tool_start("Diff", args_str=f"{file1} vs {file2}")
    else:
        log_tool_start("Diff", args_str=file1)

    try:
        if file2:
            # Two-file diff using diff command
            path1 = Path(file1).expanduser().resolve()
            path2 = Path(file2).expanduser().resolve()

            if not path1.exists():
                log_tool_error(f"File not found: {file1}")
                return f"ERROR: File not found: {file1}"

            if not path2.exists():
                log_tool_error(f"File not found: {file2}")
                return f"ERROR: File not found: {file2}"

            cmd = ["diff"]
            if unified:
                cmd.extend(["-u", f"-U{context_lines}"])

            cmd.extend([str(path1), str(path2)])

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            # diff returns 0 if identical, 1 if different, 2 if error
            if result.returncode == 2:
                log_tool_error("Diff command failed")
                return f"ERROR: {result.stderr}"
            elif result.returncode == 0:
                log_tool_complete("Files are identical")
                return "Files are identical (no differences)"
            else:
                # Has differences
                lines = result.stdout.count('\n')
                log_tool_complete(f"Found differences ({lines} lines)")
                return result.stdout

        else:
            # Git diff for single file
            path = Path(file1).expanduser().resolve()

            if not path.exists():
                log_tool_error(f"File not found: {file1}")
                return f"ERROR: File not found: {file1}"

            cmd = ["git", "diff"]
            if context_lines != 3:
                cmd.append(f"-U{context_lines}")

            cmd.append(str(path))

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)

            if result.returncode != 0:
                log_tool_error("Git diff failed")
                return f"ERROR: Git diff failed\n{result.stderr}"

            if not result.stdout.strip():
                log_tool_complete("No changes (file matches git HEAD)")
                return "No changes detected (file matches git HEAD)"

            lines = result.stdout.count('\n')
            log_tool_complete(f"Found changes ({lines} lines)")
            return result.stdout

    except subprocess.TimeoutExpired:
        log_tool_error("Timeout")
        return "ERROR: Diff operation timed out (>10s)"
    except FileNotFoundError as e:
        if "git" in str(e):
            log_tool_error("Git not available")
            return "ERROR: Git not available. For two-file diff, provide both file1 and file2 arguments"
        else:
            log_tool_error("diff command not available")
            return "ERROR: diff command not available on this system"
    except Exception as e:
        log_tool_error(str(e))
        return f"ERROR: {e}"
