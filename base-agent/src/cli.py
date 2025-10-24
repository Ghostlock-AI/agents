"""
CLI - Enhanced command-line interface for agent interaction.

Supports:
    - Interactive TUI with prompt_toolkit (multiline, path completion)
    - File attachments via /file command
    - Rich markdown rendering
    - One-shot mode: python main.py "your question"
    - Session-based conversation tracking

Commands:
    /file PATH        attach a local file to this message (with path completion)
    /enter send       toggle to send-on-enter mode
    /enter newline    toggle to newline-on-enter mode
    /quit             exit the application
"""

from __future__ import annotations

import os
import sys
from typing import List, Tuple
import shutil

try:
    from rich.console import Console
    from rich.markdown import Markdown
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    Console = None
    Markdown = None

console = Console() if HAS_RICH else None


def _max_backtick_run(s: str) -> int:
    """Find the longest consecutive run of backticks in a string."""
    max_run = run = 0
    for ch in s:
        if ch == "`":
            run += 1
            if run > max_run:
                max_run = run
        else:
            run = 0
    return max_run


def _choose_fence(content: str) -> str:
    """Choose a backtick fence longer than any backtick run in content; min 3."""
    length = max(3, _max_backtick_run(content) + 1)
    return "`" * length


def _language_from_filename(path: str) -> str:
    """Detect language from file extension."""
    name = os.path.basename(path)
    lower = name.lower()
    if lower == "dockerfile":
        return "dockerfile"
    ext = os.path.splitext(lower)[1]
    return {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".sh": "bash",
        ".bash": "bash",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".md": "markdown",
        ".txt": "text",
        ".go": "go",
        ".rs": "rust",
        ".java": "java",
        ".c": "c",
        ".cpp": "cpp",
        ".toml": "toml",
        ".ini": "ini",
    }.get(ext, "")


def _read_text_file(path: str) -> Tuple[str, int]:
    """Read a text file and return (content, byte_length)."""
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        data = fh.read()
    return data, len(data.encode("utf-8"))


def _strip_quotes(s: str) -> str:
    """Remove surrounding quotes from a string."""
    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
        return s[1:-1]
    return s


def _resolve_path(p: str) -> str:
    """Resolve a path with variable expansion and user home expansion."""
    p = _strip_quotes(p.strip())
    p = os.path.expanduser(p)
    p = os.path.expandvars(p)
    return os.path.abspath(p)


def _parse_file_commands(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    """Extract /file PATH lines and return (clean_text, attachments).

    Args:
        text: Input text that may contain /file commands

    Returns:
        Tuple of (cleaned_text, list_of_attachments)
        where each attachment is (file_path, file_content)
    """
    attachments: List[Tuple[str, str]] = []
    kept_lines: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("/file"):
            path = None
            if line.startswith("/file:"):
                path = _resolve_path(line.split(":", 1)[1])
            else:
                parts = raw_line.split(None, 1)
                if len(parts) == 2:
                    path = _resolve_path(parts[1])
            if path:
                try:
                    content, _ = _read_text_file(path)
                    attachments.append((path, content))
                except OSError as e:
                    kept_lines.append(raw_line)
                    kept_lines.append(f"[client] Failed to read {path}: {e}")
                continue
        kept_lines.append(raw_line)
    clean_text = "\n".join(kept_lines)
    return clean_text, attachments


def _build_message(prompt: str, attachments: List[Tuple[str, str]]) -> str:
    """Build final message including file attachments in code blocks."""
    if not attachments:
        return prompt
    parts = [prompt, ""]
    for path, content in attachments:
        fence = _choose_fence(content)
        lang = _language_from_filename(path)
        header = f"[FILE: {path}]"
        if not content.endswith("\n"):
            content = content + "\n"
        if lang:
            block = f"{header}\n{fence}{lang}\n{content}{fence}"
        else:
            block = f"{header}\n{fence}\n{content}{fence}"
        parts.append(block)
    return "\n\n".join(parts)


def stream_response(agent, prompt: str, session_id: str) -> None:
    """Stream response from agent and render with rich markdown."""
    accumulated = ""

    status_msg = "[blue]Thinking..." if HAS_RICH else "Thinking..."
    if HAS_RICH:
        with console.status(status_msg, spinner="dots"):
            try:
                for chunk in agent.stream(prompt, session_id):
                    if chunk:
                        accumulated += chunk
            except Exception as e:
                if HAS_RICH:
                    console.print(f"[red]Error: {e}[/red]")
                else:
                    print(f"Error: {e}")
                return
    else:
        try:
            for chunk in agent.stream(prompt, session_id):
                if chunk:
                    accumulated += chunk
        except Exception as e:
            print(f"Error: {e}")
            return

    # Render accumulated response
    if accumulated.strip():
        if HAS_RICH:
            md = Markdown(accumulated)
            console.print(md)
        else:
            print(accumulated)

    if HAS_RICH:
        console.print()  # Extra newline for spacing
    else:
        print()


def _run_tui(agent, session_id: str) -> None:
    """Run a prompt_toolkit TUI with multiline input and file attachment support."""
    try:
        from prompt_toolkit.shortcuts import PromptSession
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.styles import Style
        from prompt_toolkit.completion import NestedCompleter, PathCompleter
        from prompt_toolkit.patch_stdout import patch_stdout
    except ImportError:
        print("prompt_toolkit not installed. Run: pip install prompt_toolkit")
        _run_simple_repl(agent, session_id)
        return

    # Enter key behavior: default send-on-enter; can toggle to newline-on-enter
    send_on_enter = [True]

    # Build completer
    try:
        completer = NestedCompleter.from_nested_dict({
            "/file": PathCompleter(expanduser=True),
            "/enter": {"send": None, "newline": None},
            "/quit": None,
        })
    except AttributeError:
        try:
            completer = NestedCompleter({
                "/file": PathCompleter(expanduser=True),
                "/enter": {"send": None, "newline": None},
                "/quit": None,
            })
        except Exception:
            from prompt_toolkit.completion import WordCompleter
            completer = WordCompleter(["/file", "/enter", "/quit"])

    # Style for the left bar
    style = Style.from_dict({
        "blockquote.prefix": "ansibrightblue",
        "rule": "ansibrightblack",
    })

    bar_tokens = [("class:blockquote.prefix", "▌ ")]

    def prompt_continuation(width: int, line_number: int, is_soft_wrap: bool):
        return bar_tokens

    # Enable terminal key protocols
    def _term_write(seq: str) -> None:
        try:
            sys.stdout.write(seq)
            sys.stdout.flush()
        except Exception:
            pass

    def _enable_modified_keys() -> None:
        _term_write("\x1b[>1u")      # Enable CSI u key reporting
        _term_write("\x1b[>4;2m")    # Enable modifyOtherKeys level 2

    def _disable_modified_keys() -> None:
        _term_write("\x1b[>0u")      # Disable CSI u key reporting
        _term_write("\x1b[>4;0m")    # Disable modifyOtherKeys

    _enable_modified_keys()

    # Key bindings
    kb = KeyBindings()

    @kb.add("enter")
    def _(event):
        buf = event.app.current_buffer
        doc = buf.document
        if send_on_enter[0]:
            if doc.text.strip() == "":
                return
            buf.validate_and_handle()
        else:
            at_end = doc.cursor_position == len(doc.text)
            blank_line = doc.current_line.strip() == ""
            has_content = doc.text.strip() != ""
            if at_end and blank_line and has_content and doc.text.endswith("\n"):
                buf.validate_and_handle()
            else:
                buf.insert_text("\n")

    @kb.add("c-j")
    def _(event):
        event.current_buffer.insert_text("\n")

    for key_name in ("s-enter", "s-return"):
        try:
            @kb.add(key_name)
            def _(event):
                event.current_buffer.insert_text("\n")
        except Exception:
            pass

    for key_name in ("a-enter", "c-enter", "a-return", "c-return"):
        try:
            @kb.add(key_name)
            def _(event):
                event.current_buffer.insert_text("\n")
        except Exception:
            pass

    @kb.add("c-s")
    def _(event):
        event.app.current_buffer.validate_and_handle()

    @kb.add("c-q")
    def _(event):
        event.app.exit(exception=EOFError)

    @kb.add("c-c")
    def _(event):
        event.app.exit(exception=KeyboardInterrupt)

    session = PromptSession(style=style, completer=completer)

    try:
        while True:
            with patch_stdout():
                try:
                    text = session.prompt(
                        bar_tokens,
                        multiline=True,
                        prompt_continuation=prompt_continuation,
                        complete_while_typing=True,
                        key_bindings=kb,
                    )
                except EOFError:
                    break

            if text is None:
                break
            if text.strip() in {"/quit", "/exit"}:
                break

            # Ignore blank submits
            if text.strip() == "":
                continue

            # Handle /enter command
            ts = text.strip()
            if ts.startswith("/enter"):
                parts = ts.split(None, 1)
                if len(parts) == 2:
                    arg = parts[1].strip().lower()
                    if arg in {"send", "newline"}:
                        send_on_enter[0] = (arg == "send")
                        mode = "send-on-enter" if send_on_enter[0] else "newline-on-enter"
                        print(f"[client] Enter mode: {mode}")
                        continue
                print("[client] Usage: /enter send | /enter newline")
                continue

            # Draw separator before response
            width = shutil.get_terminal_size(fallback=(80, 20)).columns
            print("-" * max(20, min(120, width)))

            # Parse file attachments and build message
            clean, attachments = _parse_file_commands(text)
            message = _build_message(clean, attachments)

            # Stream response
            stream_response(agent, message, session_id)

            # Draw separator after response
            width = shutil.get_terminal_size(fallback=(80, 20)).columns
            print("-" * max(20, min(120, width)))

    except KeyboardInterrupt:
        raise
    finally:
        _disable_modified_keys()


def _run_simple_repl(agent, session_id: str) -> None:
    """Fallback simple REPL without prompt_toolkit."""
    print("\nBase Agent - Interactive Chat")
    print("Type 'exit', 'quit', 'q' to quit. Commands: /file <path>, /quit")
    print("-" * 60)
    print()

    while True:
        try:
            text = input("You: ").strip()
        except EOFError:
            break
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break

        if not text:
            continue

        if text.lower() in {"/quit", "/exit", "exit", "quit", "q"}:
            print("\nGoodbye!")
            break

        # Handle /enter command
        if text.startswith("/enter"):
            print("[client] /enter not supported in simple mode")
            continue

        # Parse file attachments
        clean, attachments = _parse_file_commands(text)
        message = _build_message(clean, attachments)

        # Stream response
        stream_response(agent, message, session_id)


def run_interactive(agent, session_id: str = "main_session") -> None:
    """Run interactive CLI with TUI or simple REPL fallback."""
    try:
        _run_tui(agent, session_id)
    except Exception as e:
        print(f"TUI error: {e}")
        print("Falling back to simple REPL...\n")
        _run_simple_repl(agent, session_id)
    except KeyboardInterrupt:
        print("\n\nGoodbye!")


def run_once(agent, prompt: str, session_id: str = "main_session") -> None:
    """Run a single query and exit."""
    stream_response(agent, prompt, session_id)


def run(agent, args: List[str] = None) -> None:
    """Main entry point for CLI.

    Args:
        agent: The Agent instance to use
        args: Command line arguments (default: sys.argv[1:])
    """
    if args is None:
        args = sys.argv[1:]

    try:
        if args:
            # One-shot mode: process all arguments as a single query
            prompt = " ".join(args)
            run_once(agent, prompt)
        else:
            # Interactive mode
            run_interactive(agent)
    except KeyboardInterrupt:
        # Exit with code 130 to indicate SIGINT
        try:
            sys.exit(130)
        except Exception:
            pass
