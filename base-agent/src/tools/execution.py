"""
Execution tool: Bash with streaming support

For running shell commands (compilation, git, npm, programs).
"""

import subprocess
import sys
from langchain_core.tools import tool
from tool_logger import log_tool_start, log_tool_complete, log_tool_error


@tool
def bash_exec(command: str, timeout: int = 30, stream_output: bool = False) -> str:
    """Execute shell command (for git, npm, gcc, running programs).

    Args:
        command: Shell command to execute
        timeout: Max execution time in seconds
        stream_output: If True, stream output to user in real-time

    Returns:
        Command output (stdout + stderr)

    When to use:
    - Compilation: gcc, make, cargo build
    - Version control: git commit, git push
    - Package management: npm install, pip install
    - Running programs: ./hello, python script.py

    When NOT to use:
    - Reading files: Use read_file instead
    - Searching code: Use grep_code instead
    - Finding files: Use find_files instead

    Examples:
        bash_exec("gcc hello.c -o hello")
        bash_exec("./hello")
        bash_exec("git commit -m 'feat: add greeting'")
    """
    # Truncate command for logging if too long
    cmd_display = command if len(command) < 60 else command[:57] + "..."
    log_tool_start("Bash", args_str=f'"{cmd_display}"')

    try:
        if stream_output:
            # Stream output to console in real-time
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            output_lines = []
            for line in iter(process.stdout.readline, ''):
                if line:
                    # Print to stderr with dim styling (won't interfere with tool output)
                    print(f"\033[2m  {line.rstrip()}\033[0m", file=sys.stderr, flush=True)
                    output_lines.append(line)

            process.wait(timeout=timeout)
            output = ''.join(output_lines)

            # Include return code in output
            if process.returncode != 0:
                output += f"\n[Command exited with code {process.returncode}]"

        else:
            # Standard execution
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            output = result.stdout + result.stderr

            # Include return code if non-zero
            if result.returncode != 0:
                output += f"\n[Command exited with code {result.returncode}]"

        # Log completion
        if output.strip():
            lines = output.count('\n')
            if process.returncode != 0 if stream_output else result.returncode != 0:
                log_tool_error(f"Exit code {process.returncode if stream_output else result.returncode}")
            else:
                log_tool_complete(f"Completed ({lines} lines output)")
        else:
            log_tool_complete("Completed with no output")

        return output if output.strip() else "[Command completed with no output]"

    except subprocess.TimeoutExpired:
        log_tool_error(f"Timeout after {timeout}s")
        return f"ERROR: Command timed out after {timeout}s"
    except Exception as e:
        log_tool_error(str(e))
        return f"ERROR executing command: {e}"
