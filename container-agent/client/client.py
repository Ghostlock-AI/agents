"""
Usage:
    python client.py                 # TUI multiline client (prompt_toolkit)
    python client.py "hello world"   # one-shot message

TUI commands:
    /file PATH   attach a local file to this message (with path completion)
    /quit        exit

Environment:
    SERVER: server URL (default http://localhost:8000/chat)
    THREAD_ID: conversation thread id (default random)
"""
from __future__ import annotations

import os
import sys
from uuid import uuid4
from typing import List, Tuple
import threading
import time
import itertools
import shutil

import httpx
from rich.console import Console
from rich.markdown import Markdown

SERVER = os.getenv("SERVER", "http://localhost:8000/chat")
THREAD_ID = os.getenv("THREAD_ID", str(uuid4()))

console = Console()


def stream_once(prompt: str) -> None:
    accumulated = ""

    with console.status("[blue]Thinking...", spinner="dots"):
        try:
            with httpx.stream(
                "POST",
                SERVER,
                json={"message": prompt, "thread_id": THREAD_ID},
                timeout=None,
            ) as r:
                r.raise_for_status()
                for chunk in r.iter_text():
                    if chunk:
                        accumulated += chunk
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return

    # Render accumulated response as markdown
    if accumulated.strip():
        md = Markdown(accumulated)
        console.print(md)
    console.print()  # Extra newline for spacing


def _max_backtick_run(s: str) -> int:
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
    # Choose a backtick fence longer than any backtick run in content; min 3
    length = max(3, _max_backtick_run(content) + 1)
    return "`" * length


def _language_from_filename(path: str) -> str:
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
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        data = fh.read()
    return data, len(data.encode("utf-8"))


def _strip_quotes(s: str) -> str:
    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
        return s[1:-1]
    return s


def _resolve_path(p: str) -> str:
    p = _strip_quotes(p.strip())
    p = os.path.expanduser(p)
    p = os.path.expandvars(p)
    return os.path.abspath(p)


def _parse_file_commands(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    """Extract /file PATH lines and return (clean_text, attachments)."""
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


def _run_tui() -> None:
    """Run a prompt_toolkit TUI that preserves classic I/O sequence.

    - Draw a left "▌ " bar for the live prompt and echoed input.
    - Enter submits; Shift+Enter/Ctrl-J insert newline.
    - Ctrl-C/Ctrl-Q quit. Path completion for /file.
    - Uses PromptSession so previous LLM output stays on screen; next prompt
      appears on the line after, not overwriting prior output.
    """
    try:
        from prompt_toolkit.shortcuts import PromptSession
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.styles import Style
        from prompt_toolkit.completion import NestedCompleter, PathCompleter
        from prompt_toolkit.patch_stdout import patch_stdout
    except Exception:
        print("prompt_toolkit not installed. Run: pip install prompt_toolkit httpx")
        # Fallback to simple REPL
        while True:
            try:
                line = input("You: ")
            except EOFError:
                break
            if not line:
                continue
            if line.strip() in {"/quit", "/exit"}:
                break
            stream_once(line)
        return

    # Enter key behavior: default send-on-enter; can toggle to newline-on-enter.
    send_on_enter = [True]  # list for mutability in closures

    # Build completer (handle prompt_toolkit version differences)
    try:
        completer = NestedCompleter.from_nested_dict({
            "/file": PathCompleter(expanduser=True),
            "/enter": {"send": None, "newline": None},
            "/quit": None,
        })
    except AttributeError:
        try:
            # Older versions may accept dict in constructor
            completer = NestedCompleter({
                "/file": PathCompleter(expanduser=True),
                "/enter": {"send": None, "newline": None},
                "/quit": None,
            })
        except Exception:
            from prompt_toolkit.completion import WordCompleter
            completer = WordCompleter(["/file", "/enter", "/quit"])  # minimal fallback

    # Style for the left bar
    style = Style.from_dict({
        "blockquote.prefix": "ansibrightblue",
        "rule": "ansibrightblack",
    })

    # Prompt prefix and continuation show the same left bar
    bar_tokens = [("class:blockquote.prefix", "▌ ")]

    def prompt_continuation(width: int, line_number: int, is_soft_wrap: bool):
        return bar_tokens

    # Try to enable terminal key-reporting protocols so Shift/Alt/Ctrl modified
    # Enter can be distinguished without user configuration in many terminals
    # (iTerm2, kitty, xterm). These are no-ops on unsupported terminals.
    def _term_write(seq: str) -> None:
        try:
            sys.stdout.write(seq)
            sys.stdout.flush()
        except Exception:
            pass

    def _enable_modified_keys() -> None:
        # kitty keyboard protocol (CSI u) and xterm modifyOtherKeys (v2)
        # If unsupported, terminals ignore silently.
        _term_write("\x1b[>1u")      # Enable CSI u key reporting
        _term_write("\x1b[>4;2m")    # Enable modifyOtherKeys level 2

    def _disable_modified_keys() -> None:
        _term_write("\x1b[>0u")      # Disable CSI u key reporting
        _term_write("\x1b[>4;0m")    # Disable modifyOtherKeys

    _enable_modified_keys()

    # Key bindings
    kb = KeyBindings()

    # Enter behavior: depending on mode, submit or insert newline. In newline mode,
    # double-Enter at end (on an empty line) submits if there is non-whitespace content.
    @kb.add("enter")
    def _(event):
        buf = event.app.current_buffer
        doc = buf.document
        if send_on_enter[0]:
            # Ignore submission on blank input; keep the prompt active
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

    # Newline on Ctrl-J
    @kb.add("c-j")
    def _(event):
        event.current_buffer.insert_text("\n")

    # Best-effort Shift+Enter (and variants) for newline. Terminals differ in naming.
    for key_name in ("s-enter", "s-return"):
        try:
            @kb.add(key_name)
            def _(event):
                event.current_buffer.insert_text("\n")
        except Exception:
            pass

    # Optional Alt+Enter and Ctrl+Enter newline fallbacks
    for key_name in ("a-enter", "c-enter", "a-return", "c-return"):
        try:
            @kb.add(key_name)
            def _(event):
                event.current_buffer.insert_text("\n")
        except Exception:
            pass

    # Submit on Ctrl-S (handy on some keyboards)
    @kb.add("c-s")
    def _(event):
        event.app.current_buffer.validate_and_handle()

    # Quit on Ctrl-Q / Ctrl-C
    @kb.add("c-q")
    def _(event):
        event.app.exit(exception=EOFError)

    @kb.add("c-c")
    def _(event):
        # Raise KeyboardInterrupt to terminate the program
        event.app.exit(exception=KeyboardInterrupt)

    session = PromptSession(style=style, completer=completer)

    # No intro print; keep the interface clean.
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
            # Ignore blank submits entirely (no separators, no request)
            if text.strip() == "":
                continue
            # Local client commands (not sent to server)
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

            # Print a horizontal rule before the response
            width = shutil.get_terminal_size(fallback=(80, 20)).columns
            print("-" * max(20, min(120, width)))

            clean, attachments = _parse_file_commands(text)
            message = _build_message(clean, attachments)
            stream_once(message)
            # Draw a trailing rule after the LLM response
            width = shutil.get_terminal_size(fallback=(80, 20)).columns
            print("-" * max(20, min(120, width)))
    except KeyboardInterrupt:
        # Propagate to main for a clean process exit code.
        raise
    finally:
        _disable_modified_keys()


def main() -> None:
    try:
        if len(sys.argv) > 1:
            stream_once(" ".join(sys.argv[1:]))
            return
        _run_tui()
    except KeyboardInterrupt:
        # Exit with code 130 to indicate SIGINT, no traceback noise.
        try:
            import sys as _sys
            _sys.exit(130)
        except Exception:
            pass


if __name__ == "__main__":
    main()
