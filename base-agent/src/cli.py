"""
CLI - Enhanced command-line interface for agent interaction.

Supports:
    - Interactive TUI with prompt_toolkit (multiline, path completion)
    - File attachments via /file command
    - Rich markdown rendering
    - One-shot mode: python main.py "your question"
    - Session-based conversation tracking
    - Runtime reasoning strategy switching

Commands:
    /file PATH                attach a local file to this message (with path completion)
    /enter send               toggle to send-on-enter mode
    /enter newline            toggle to newline-on-enter mode
    /reasoning list           list all available reasoning strategies
    /reasoning current        show current reasoning strategy
    /reasoning switch NAME    switch to a different reasoning strategy
    /reasoning info [NAME]    show detailed info about a strategy
    /quit                     exit the application
"""

from __future__ import annotations

import os
import sys
from typing import List, Tuple
import shutil
import time
from pathlib import Path
import threading
import re
import glob
import math

try:
    from rich.console import Console
    from rich.markdown import Markdown
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.live import Live
    from rich.spinner import SPINNERS
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    Console = None
    Markdown = None

console = Console() if HAS_RICH else None

# Banner sizing (can override with env var)
BANNER_MAX_WIDTH = None
try:
    _bw = os.getenv("BANNER_MAX_WIDTH")
    if _bw:
        BANNER_MAX_WIDTH = max(40, min(120, int(_bw)))
except Exception:
    BANNER_MAX_WIDTH = None

# Persistent mascot toggle
PERSISTENT_MASCOT = (os.getenv("PERSISTENT_MASCOT", "1").strip() not in {"0", "false", "False", "no", ""})

# Optional: OpenCV for PNG -> ASCII conversion
try:
    import cv2  # type: ignore
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

# Simple 2-frame slime animation - hardcoded ASCII art
SLIME_FRAMES = [
    # Frame 1 - normal
    """
       @@@@@
     @@@@@@@@@
    @@@@@@@@@@@
   @@@@@@@@@@@@@
    @@@@@@@@@@@
     @@@@@@@@@
       @@@@@
    """,
    # Frame 2 - squished (slight variation)
    """
      @@@@@@@
    @@@@@@@@@@@
   @@@@@@@@@@@@@
  @@@@@@@@@@@@@@@
   @@@@@@@@@@@@@
    @@@@@@@@@@@
      @@@@@@@
    """
]


# Simple banner helpers from simple_banner.py (legacy)
def image_to_ascii(image_path: str, width: int = 40) -> str:
    """Convert image to ASCII using Pillow (if available).

    Returns an empty string if Pillow is not installed or on failure.
    """
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return ""
    try:
        img = Image.open(image_path).convert('L')

        # Resize with approximate character aspect ratio
        aspect_ratio = img.height / img.width if img.width else 1.0
        height = int(width * aspect_ratio * 0.45)
        height = max(1, height)
        img = img.resize((width, height))

        chars = " .:-=+*#%@"
        ascii_art_lines: List[str] = []
        for y in range(img.height):
            row = []
            for x in range(img.width):
                pixel = img.getpixel((x, y))
                idx = min(int(pixel / 255 * (len(chars) - 1)), len(chars) - 1)
                row.append(chars[idx])
            ascii_art_lines.append("".join(row))
        return "\n".join(ascii_art_lines)
    except Exception:
        return ""


STATIC_SLIME = """
    .@@@@@.
   @@@@@@@@@
  @@@@@@@@@@@
 @@@@@@@@@@@@@
  @@@@@@@@@@@
   @@@@@@@@@
    @@@@@@@
"""


def load_frames_from_txt(directory: str) -> List[str]:
    """Load ASCII frames from text files in a directory (frame_*.txt)."""
    try:
        frame_files = sorted(glob.glob(f"{directory}/frame_*.txt"))
        frames: List[str] = []
        for f in frame_files:
            with open(f, 'r') as file:
                frames.append(file.read())
        return frames
    except Exception:
        return []


def show_animated_banner(text_content: str, ascii_frames: List[str] | None = None, duration: float = 3.0) -> None:
    """Show banner with optional animation (legacy helper)."""
    if not HAS_RICH:
        # Fallback text-only
        _display_static_banner(text_content)
        return

    if not ascii_frames:
        # Show static simple slime + text using our compact panel
        panel = Panel(text_content, border_style="blue")
        console.print(panel)
        return

    fps = 12
    with Live(console=console, refresh_per_second=fps) as live:
        start = time.time()
        while time.time() - start < duration:
            frame_idx = int((time.time() - start) * fps) % len(ascii_frames)
            ascii_art = ascii_frames[frame_idx]
            panel = _make_compact_banner(ascii_art, text_content)
            live.update(panel)
            time.sleep(1 / fps)

def _make_compact_banner(left_art: str, text_content: str) -> Panel:
    """Render a single compact panel with ASCII mascot on the left and text on the right."""
    from rich.table import Table

    # Compute widths
    ascii_lines = left_art.splitlines() if left_art else []
    ascii_width = max((len(ln) for ln in ascii_lines), default=0)

    term_width = shutil.get_terminal_size(fallback=(80, 24)).columns
    # Keep things small and readable
    default_cap = 60
    cap = BANNER_MAX_WIDTH if BANNER_MAX_WIDTH is not None else default_cap
    max_panel = min(cap, max(46, term_width - 8))
    # Allocate space to text after the mascot and a small gap
    text_target = max_panel - (ascii_width + 3)
    if text_target < 24:
        # if terminal is very narrow, reduce mascot width effect
        text_target = 24

    # Build a compact grid with fixed first column
    table = Table.grid(padding=(0, 1))
    table.expand = False
    if ascii_width > 0:
        table.add_column(no_wrap=True, width=ascii_width)
    else:
        table.add_column(no_wrap=True)
    table.add_column(no_wrap=False, width=text_target)
    table.add_row(left_art, text_content)

    panel = Panel(
        table,
        border_style="blue",
        title="[blue]Base Agent[/blue]",
        width=max_panel,
    )
    return panel


# Persistent mascot animation management
_mascot_thread: threading.Thread | None = None
_mascot_stop: threading.Event | None = None


def _start_persistent_mascot(agent) -> None:
    """Start a background thread that keeps the mascot animating above the prompt.

    Pauses during streaming and resumes afterwards. Uses PNG frames when available,
    falls back to built-in ASCII frames otherwise.
    """
    global _mascot_thread, _mascot_stop
    if not HAS_RICH:
        return
    if _mascot_thread and _mascot_thread.is_alive():
        return

    _mascot_stop = threading.Event()

    def _runner():
        # Prepare frames: prefer pre-generated ASCII, else PNGs, else built-in
        frames = _load_ascii_text_frames(directory="ascii_frames", max_frames=10) \
            or _load_png_mascot_frames(max_frames=2, width=20) \
            or SLIME_FRAMES
        if not frames:
            return
        fps = 3
        frame_delay = 1.0 / fps

        try:
            with Live(console=console, refresh_per_second=fps) as live:
                i = 0
                while not _mascot_stop.is_set():  # type: ignore[attr-defined]
                    # Recompute text to reflect current strategy/model
                    model_name = agent.model_name
                    current_strategy = agent.get_current_strategy_name()
                    strategy_info = agent.get_strategy_info(current_strategy)
                    text_content = f"""[magenta]agent:[/] Base Agent\n[magenta]model:[/] {model_name}\n[magenta]planning:[/] {current_strategy.upper()} - {strategy_info['description'].split('.')[0]}\n[magenta]tools:[/]\n  - internet search (DuckDuckGo)\n  - web fetch (HTTP/HTTPS)\n  - command line (shell)"""

                    ascii_art = frames[i % len(frames)]
                    panel = _make_compact_banner(ascii_art, text_content)
                    live.update(panel)
                    i += 1
                    # Sleep in small increments to react to stop quickly
                    end_time = time.time() + frame_delay
                    while time.time() < end_time:
                        if _mascot_stop.is_set():
                            break
                        time.sleep(0.02)
        except Exception:
            # Best-effort; don't crash REPL on animation issues
            pass

    _mascot_thread = threading.Thread(target=_runner, name="mascot", daemon=True)
    _mascot_thread.start()


def _stop_persistent_mascot() -> None:
    """Stop the background mascot animation if running."""
    global _mascot_thread, _mascot_stop
    if _mascot_stop is not None:
        _mascot_stop.set()
    if _mascot_thread is not None:
        try:
            _mascot_thread.join(timeout=1.0)
        except Exception:
            pass
    _mascot_thread = None
    _mascot_stop = None


def _image_to_ascii_cv2(image_path: str, width: int = 20) -> str:
    """Convert an image to ASCII art using OpenCV (no Pillow dependency).

    Args:
        image_path: Path to the PNG/JPG image.
        width: Target character width for ASCII rendering.

    Returns:
        ASCII art string for the image.
    """
    if not HAS_CV2:
        return ""

    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return ""

    # Split alpha if present
    alpha = None
    if len(img.shape) == 3 and img.shape[2] == 4:
        b, g, r, a = cv2.split(img)
        gray = cv2.cvtColor(cv2.merge((b, g, r)), cv2.COLOR_BGR2GRAY)
        alpha = a
    elif len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    h, w = gray.shape[:2]
    if w == 0:
        return ""

    # Account for character aspect ratio (roughly 2:1 height:width)
    target_height = max(1, int((h / w) * width * 0.5))
    resized = cv2.resize(gray, (width, target_height), interpolation=cv2.INTER_AREA)
    resized_alpha = None
    if alpha is not None:
        resized_alpha = cv2.resize(alpha, (width, target_height), interpolation=cv2.INTER_AREA)

    # Denser ramp for better detail
    chars = " .,:;i1tfLCG08@"
    buckets = len(chars) - 1

    lines: List[str] = []
    for y in range(resized.shape[0]):
        row_chars = []
        for x in range(resized.shape[1]):
            # Transparent -> space
            if resized_alpha is not None and int(resized_alpha[y, x]) < 16:
                row_chars.append(" ")
                continue

            px = int(resized[y, x])
            idx = min(buckets, (px * buckets) // 255)
            row_chars.append(chars[idx])
        lines.append("".join(row_chars))

    return "\n".join(lines)


def _load_png_mascot_frames(max_frames: int = 2, width: int = 20) -> List[str]:
    """Search for PNG mascot frames at repo root and convert to ASCII.

    Looks for files named like slime*.png next to the project root.

    Args:
        max_frames: Maximum number of frames to use for animation.
        width: ASCII width per frame.

    Returns:
        List of ASCII frame strings (possibly length 0 or 1).
    """
    if not HAS_CV2:
        return []

    try:
        base_dir = Path(__file__).resolve().parent.parent
        candidates = sorted(p for p in base_dir.glob("slime*.png") if p.is_file())
        if not candidates:
            return []

        frames: List[str] = []
        for p in candidates[:max_frames]:
            ascii_frame = _image_to_ascii_cv2(str(p), width=width)
            if ascii_frame:
                frames.append(ascii_frame)

        return frames
    except Exception:
        return []


def _load_ascii_text_frames(directory: str = "ascii_frames", max_frames: int = 10) -> List[str]:
    """Load pre-generated ASCII frames from a directory (frame_*.txt)."""
    try:
        base_dir = Path(__file__).resolve().parent.parent / directory
        files = sorted(base_dir.glob("frame_*.txt"))[:max_frames]
        frames: List[str] = []
        for fp in files:
            frames.append(fp.read_text())
        return frames
    except Exception:
        return []


def display_startup_banner(agent, animate: bool = True, duration: float = 3.0) -> None:
    """Display a simple text-only startup banner with agent info."""
    # Get agent info
    model_name = agent.model_name
    current_strategy = agent.get_current_strategy_name()
    strategy_info = agent.get_strategy_info(current_strategy)

    # Format the text content with low-contrast keywords and default-color values
    # Use light grey for low-contrast parts (works on dark and light themes)
    kstyle = "color(245)"  # light grey
    cwd = os.getcwd()
    planning_line = f"{current_strategy.upper()} - {strategy_info['description'].split('.')[0]}"

    text_content = (
        f"[{kstyle}]agent:[/] Base Agent\n"
        f"[{kstyle}]model:[/] {model_name}\n"
        f"[{kstyle}]planning:[/] {planning_line}\n"
        f"[{kstyle}]tools:[/]\n"
        f"  - internet search (DuckDuckGo)\n"
        f"  - web fetch (HTTP/HTTPS)\n"
        f"  - command line (shell)\n"
        f"[{kstyle}]directory:[/] {cwd}"
    )

    if HAS_RICH:
        # Text-only panel, auto-widen to fit content and planning line
        def _strip_rich_tags(s: str) -> str:
            # Remove [tag] and [/tag] patterns for width calc
            return re.sub(r"\[[^\]]*\]", "", s)

        visible = _strip_rich_tags(text_content)
        longest = max((len(line) for line in visible.splitlines()), default=0)

        term_width = shutil.get_terminal_size(fallback=(80, 24)).columns
        # Base desired width adds padding for borders, keep it reasonably wide
        desired = longest + 4
        min_w = 72
        default_cap = min(max(term_width - 4, min_w), 120)
        cap = BANNER_MAX_WIDTH if BANNER_MAX_WIDTH is not None else default_cap
        width_target = min(max(desired, min_w), cap)

        panel = Panel(text_content, border_style="color(240)", width=width_target)
        console.print()
        console.print(panel)
        return
    else:
        # Fallback for non-rich display
        _display_static_banner(text_content)


def _display_static_banner(text_content: str) -> None:
    """Display static banner without animation."""
    if HAS_RICH:
        width_cap = BANNER_MAX_WIDTH if BANNER_MAX_WIDTH is not None else 60
        panel = Panel(text_content, border_style="color(240)", width=width_cap)
        console.print()
        console.print(panel)
    else:
        # Fallback for non-rich display
        print()
        print("=" * 60)
        print("agent: Base Agent")
        print(f"model: {text_content.split('model:[/] ')[1].split('[')[0].strip()}")
        print("tools:")
        print("  - internet search (DuckDuckGo)")
        print("  - web fetch (HTTP/HTTPS)")
        print("  - command line (shell)")
        print("=" * 60)


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


def handle_reasoning_command(agent, args: List[str]) -> bool:
    """
    Handle /reasoning commands.

    Args:
        agent: The agent instance
        args: Command arguments (e.g., ['list'], ['switch', 'rewoo'])

    Returns:
        True if command was handled, False otherwise
    """
    if not args:
        print("[client] Usage: /reasoning <list|current|switch|info>")
        return True

    subcommand = args[0].lower()

    if subcommand == "list":
        # List all available strategies
        strategies = agent.list_strategies()

        if HAS_RICH:
            console.print("\n[bold cyan]Available Reasoning Strategies:[/bold cyan]\n")
            for strategy in strategies:
                current_marker = "[green]✓[/green] " if strategy["is_current"] else "  "
                console.print(f"{current_marker}[bold]{strategy['name']}[/bold]")
                console.print(f"  {strategy['description']}\n")
        else:
            print("\nAvailable Reasoning Strategies:\n")
            for strategy in strategies:
                current_marker = "✓ " if strategy["is_current"] else "  "
                print(f"{current_marker}{strategy['name']}")
                print(f"  {strategy['description']}\n")

        return True

    elif subcommand == "current":
        # Show current strategy
        current = agent.get_current_strategy_name()
        info = agent.get_strategy_info()

        if HAS_RICH:
            console.print(f"\n[bold cyan]Current Strategy:[/bold cyan] [green]{current}[/green]")
            console.print(f"[dim]{info['description']}[/dim]\n")
        else:
            print(f"\nCurrent Strategy: {current}")
            print(f"{info['description']}\n")

        return True

    elif subcommand == "switch":
        # Switch to a different strategy
        if len(args) < 2:
            print("[client] Usage: /reasoning switch <strategy_name>")
            return True

        new_strategy = args[1].lower()

        try:
            agent.switch_reasoning_strategy(new_strategy)

            if HAS_RICH:
                console.print(f"\n[green]✓[/green] Switched to [bold]{new_strategy}[/bold] strategy\n")
            else:
                print(f"\n✓ Switched to {new_strategy} strategy\n")

            # Re-display the text-only banner
            display_startup_banner(agent, animate=False)

        except KeyError as e:
            if HAS_RICH:
                console.print(f"\n[red]Error:[/red] {e}\n")
            else:
                print(f"\nError: {e}\n")

        return True

    elif subcommand == "info":
        # Show detailed info about a strategy
        strategy_name = args[1].lower() if len(args) > 1 else None

        try:
            info = agent.get_strategy_info(strategy_name)

            if HAS_RICH:
                console.print(f"\n[bold cyan]Strategy:[/bold cyan] [bold]{info['name']}[/bold]")
                if info['is_current']:
                    console.print("[green]✓ Currently active[/green]")
                console.print(f"\n[bold]Description:[/bold]\n{info['description']}")
                console.print(f"\n[bold]Streaming Support:[/bold] {'Yes' if info['supports_streaming'] else 'No'}")

                if info['config']:
                    console.print("\n[bold]Configuration:[/bold]")
                    for key, value in info['config'].items():
                        console.print(f"  {key}: {value}")

                console.print()
            else:
                print(f"\nStrategy: {info['name']}")
                if info['is_current']:
                    print("✓ Currently active")
                print(f"\nDescription:\n{info['description']}")
                print(f"\nStreaming Support: {'Yes' if info['supports_streaming'] else 'No'}")

                if info['config']:
                    print("\nConfiguration:")
                    for key, value in info['config'].items():
                        print(f"  {key}: {value}")

                print()

        except KeyError as e:
            if HAS_RICH:
                console.print(f"\n[red]Error:[/red] {e}\n")
            else:
                print(f"\nError: {e}\n")

        return True

    else:
        print(f"[client] Unknown reasoning subcommand: {subcommand}")
        print("[client] Usage: /reasoning <list|current|switch|info>")
        return True


def stream_response(agent, prompt: str, session_id: str) -> None:
    """Stream response from agent and render with rich markdown."""
    accumulated = ""

    # Show current reasoning strategy
    current_strategy = agent.get_current_strategy_name()
    strategy_label = f"[{current_strategy.upper()}]"

    status_msg = f"[blue]{strategy_label} Thinking..." if HAS_RICH else f"{strategy_label} Thinking..."

    def _pick_spinner_name() -> str:
        if not HAS_RICH:
            return "dots"
        for name in ("squareCorners", "dots9", "dots12", "dots"):
            try:
                if name in SPINNERS:
                    return name
            except Exception:
                break
        return "dots"

    if HAS_RICH:
        with console.status(status_msg, spinner=_pick_spinner_name(), spinner_style="color(245)"):
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

    # Display simple text-only startup banner
    display_startup_banner(agent, animate=False)

    # Enter key behavior: default send-on-enter; can toggle to newline-on-enter
    send_on_enter = [True]

    # Build completer
    try:
        completer = NestedCompleter.from_nested_dict({
            "/file": PathCompleter(expanduser=True),
            "/enter": {"send": None, "newline": None},
            "/reasoning": {
                "list": None,
                "current": None,
                "switch": {
                    "react": None,
                    "rewoo": None,
                    "plan-execute": None,
                    "lats": None,
                },
                "info": {
                    "react": None,
                    "rewoo": None,
                    "plan-execute": None,
                    "lats": None,
                },
            },
            "/quit": None,
        })
    except AttributeError:
        try:
            completer = NestedCompleter({
                "/file": PathCompleter(expanduser=True),
                "/enter": {"send": None, "newline": None},
                "/reasoning": {
                    "list": None,
                    "current": None,
                    "switch": {
                        "react": None,
                        "rewoo": None,
                        "plan-execute": None,
                        "lats": None,
                    },
                    "info": {
                        "react": None,
                        "rewoo": None,
                        "plan-execute": None,
                        "lats": None,
                    },
                },
                "/quit": None,
            })
        except Exception:
            from prompt_toolkit.completion import WordCompleter
            completer = WordCompleter(["/file", "/enter", "/reasoning", "/quit"])

    # Style for the left bar
    # Use prompt_toolkit-supported color syntax (hex), not Rich-style 'color(n)'
    style = Style.from_dict({
        "blockquote.prefix": "fg:#888888",
        "rule": "fg:#888888",
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

    def _recolor_last_prompt_submission(submitted: str) -> None:
        """Best-effort recolor of the just-submitted prompt in-place to low-contrast.

        We approximate the number of terminal rows consumed (accounting for simple wrapping)
        and move the cursor up to rewrite those lines in grey. This avoids echoing a duplicate.
        """
        try:
            if not HAS_RICH:
                return
            if not submitted:
                return
            prefix = "▌ "
            width = max(10, shutil.get_terminal_size(fallback=(80, 20)).columns)
            logical_lines = submitted.splitlines() or [""]
            # Estimate how many terminal rows the submission used
            rows = 0
            for ln in logical_lines:
                length = len(prefix) + len(ln)
                rows += max(1, math.ceil(length / width))
            # Move cursor up and rewrite
            sys.stdout.write(f"\x1b[{rows}A")
            for ln in logical_lines:
                sys.stdout.write("\x1b[2K\r")  # clear line
                console.print(f"[color(245)]{prefix}{ln}[/color(245)]", end="")
                sys.stdout.write("\n")
            sys.stdout.flush()
        except Exception:
            # If anything goes wrong, quietly skip recolor
            pass

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

            # Handle /reasoning command
            if ts.startswith("/reasoning"):
                parts = ts.split()[1:]  # Get all parts after /reasoning
                handle_reasoning_command(agent, parts)
                continue

            # Recolor the just-submitted text in-place to low-contrast
            _recolor_last_prompt_submission(text)

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
    # Display startup banner
    display_startup_banner(agent)

    print("\nType 'exit', 'quit', 'q' to quit.")
    print("Commands: /file <path>, /reasoning <list|current|switch|info>, /quit")
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

        # Handle /reasoning command
        if text.startswith("/reasoning"):
            parts = text.split()[1:]  # Get all parts after /reasoning
            handle_reasoning_command(agent, parts)
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
