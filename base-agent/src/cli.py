"""
CLI - Minimal command-line interface for agent interaction.

Supports:
    - Interactive TUI with prompt_toolkit (multiline, path completion)
    - File attachments via /file command
    - Rich markdown rendering
    - One-shot mode: python main.py "your question"
    - Runtime reasoning strategy switching

Commands:
    /file PATH                attach a local file to this message
    /reasoning list           list all available reasoning strategies
    /reasoning current        show current reasoning strategy
    /reasoning switch NAME    switch to a different reasoning strategy
    /reasoning info [NAME]    show detailed info about a strategy
    /help                     show available commands
    /quit                     exit the application
"""

from __future__ import annotations

import os
import sys
from typing import List, Tuple, Callable, Dict
from dataclasses import dataclass
import shutil

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import SPINNERS

from prompt_toolkit.shortcuts import PromptSession
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from prompt_toolkit.completion import NestedCompleter, PathCompleter, FuzzyCompleter
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.filters import Condition
from prompt_toolkit.application.current import get_app

console = Console()


# ============================================================================
# Command Dispatcher
# ============================================================================

@dataclass
class CommandResult:
    """Result from command execution."""
    handled: bool = False
    should_exit: bool = False
    should_process: bool = False
    message: str = ""


class CommandDispatcher:
    """Centralized command dispatcher for CLI commands."""

    def __init__(self, agent):
        self.agent = agent
        self.commands: Dict[str, Callable] = {
            "help": self._handle_help,
            "quit": self._handle_quit,
            "exit": self._handle_quit,
            "reasoning": self._handle_reasoning,
        }

    def dispatch(self, text: str) -> CommandResult:
        """Dispatch a command and return the result."""
        text = text.strip()

        if not text:
            return CommandResult(handled=True, should_exit=False)

        if text.lower() in {"quit", "q", "exit", "/quit", "/exit"}:
            return self._handle_quit(text, [])

        if text in {"/help", "/?"}:
            return self._handle_help(text, [])

        if text.startswith("/"):
            parts = text[1:].split()
            if not parts:
                return CommandResult(handled=False, should_process=True, message=text)

            command = parts[0].lower()
            args = parts[1:]

            handler = self.commands.get(command)
            if handler:
                return handler(text, args)

        return CommandResult(handled=False, should_process=True, message=text)

    def _handle_help(self, text: str, args: List[str]) -> CommandResult:
        """Handle /help command."""
        console.print()
        console.print("[#888888]/file PATH[/#888888] — attach a local file")
        console.print("[#888888]/reasoning[/#888888] — list or switch strategy")
        console.print("[#888888]/help[/#888888] — show available commands")
        console.print("[#888888]/quit[/#888888] — exit application")
        console.print()
        return CommandResult(handled=True, should_exit=False)

    def _handle_quit(self, text: str, args: List[str]) -> CommandResult:
        """Handle /quit and /exit commands."""
        return CommandResult(handled=True, should_exit=True)

    def _handle_reasoning(self, text: str, args: List[str]) -> CommandResult:
        """Handle /reasoning commands."""
        if not args:
            console.print("[yellow]Usage: /reasoning <list|current|switch|info>[/yellow]")
            return CommandResult(handled=True, should_exit=False)

        subcommand = args[0].lower()

        if subcommand == "list":
            self._reasoning_list()
        elif subcommand == "current":
            self._reasoning_current()
        elif subcommand == "switch":
            self._reasoning_switch(args[1:])
        elif subcommand == "info":
            self._reasoning_info(args[1:])
        else:
            console.print(f"[yellow]Unknown subcommand: {subcommand}[/yellow]")
            console.print("[yellow]Usage: /reasoning <list|current|switch|info>[/yellow]")

        return CommandResult(handled=True, should_exit=False)

    def _reasoning_list(self) -> None:
        """List all available reasoning strategies."""
        strategies = self.agent.list_strategies()
        console.print("\n[bold cyan]Available Reasoning Strategies:[/bold cyan]\n")
        for strategy in strategies:
            marker = "[green]✓[/green] " if strategy["is_current"] else "  "
            console.print(f"{marker}[bold]{strategy['name']}[/bold]")
            console.print(f"  {strategy['description']}\n")

    def _reasoning_current(self) -> None:
        """Show current reasoning strategy."""
        current = self.agent.get_current_strategy_name()
        info = self.agent.get_strategy_info()
        console.print(f"\n[bold cyan]Current Strategy:[/bold cyan] [green]{current}[/green]")
        console.print(f"[dim]{info['description']}[/dim]\n")

    def _reasoning_switch(self, args: List[str]) -> None:
        """Switch to a different reasoning strategy."""
        if not args:
            console.print("[yellow]Usage: /reasoning switch <strategy_name>[/yellow]")
            return

        new_strategy = args[0].lower()
        try:
            self.agent.switch_reasoning_strategy(new_strategy)
            console.print(f"\n[green]✓[/green] Switched to [bold]{new_strategy}[/bold] strategy\n")
            display_banner(self.agent)
        except KeyError as e:
            console.print(f"\n[red]Error:[/red] {e}\n")

    def _reasoning_info(self, args: List[str]) -> None:
        """Show detailed info about a reasoning strategy."""
        strategy_name = args[0].lower() if args else None

        try:
            info = self.agent.get_strategy_info(strategy_name)
            console.print(f"\n[bold cyan]Strategy:[/bold cyan] [bold]{info['name']}[/bold]")
            if info['is_current']:
                console.print("[green]✓ Currently active[/green]")
            console.print(f"\n[bold]Description:[/bold]\n{info['description']}")
            console.print(f"\n[bold]Streaming:[/bold] {'Yes' if info['supports_streaming'] else 'No'}")

            if info['config']:
                console.print("\n[bold]Configuration:[/bold]")
                for key, value in info['config'].items():
                    console.print(f"  {key}: {value}")
            console.print()
        except KeyError as e:
            console.print(f"\n[red]Error:[/red] {e}\n")

    def get_completions(self) -> Dict:
        """Get command completions for prompt_toolkit."""
        return {
            "/file": PathCompleter(expanduser=True, only_directories=False),
            "/help": None,
            "/reasoning": {
                "list": None,
                "current": None,
                "switch": {"react": None, "rewoo": None, "plan-execute": None, "lats": None},
                "info": {"react": None, "rewoo": None, "plan-execute": None, "lats": None},
            },
            "/quit": None,
            "/exit": None,
        }


# ============================================================================
# Display Functions
# ============================================================================

def display_banner(agent) -> None:
    """Display startup banner with agent info."""
    model = agent.model_name
    strategy = agent.get_current_strategy_name()
    strategy_info = agent.get_strategy_info(strategy)
    desc = strategy_info['description'].split('.')[0]

    content = (
        f"[dim]agent:[/dim] Base Agent\n"
        f"[dim]model:[/dim] {model}\n"
        f"[dim]planning:[/dim] {strategy.upper()} - {desc}\n"
        f"[dim]tools:[/dim]\n"
        f"  - internet search (DuckDuckGo)\n"
        f"  - web fetch (HTTP/HTTPS)\n"
        f"  - command line (shell)\n"
        f"[dim]directory:[/dim] {os.getcwd()}"
    )

    # Calculate panel width based on content
    visible_text = content.replace("[dim]", "").replace("[/dim]", "")
    max_line = max(len(line) for line in visible_text.splitlines())
    width = min(max(max_line + 4, 70), 120)

    panel = Panel(content, border_style="dim", width=width)
    console.print()
    console.print(panel)


def stream_response(agent, prompt: str, session_id: str) -> None:
    """Stream response from agent (plain text output)."""
    accumulated = ""
    strategy = agent.get_current_strategy_name()
    status = f"[blue][{strategy.upper()}] Thinking...[/blue]"

    # Pick a spinner
    spinner_name = "dots"
    for name in ("squareCorners", "dots9", "dots12", "dots"):
        if name in SPINNERS:
            spinner_name = name
            break

    with console.status(status, spinner=spinner_name, spinner_style="dim"):
        try:
            for chunk in agent.stream(prompt, session_id):
                if chunk:
                    accumulated += chunk
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return

    if accumulated.strip():
        # Print as plain text instead of markdown
        console.print(accumulated)
    console.print()


# ============================================================================
# File Attachment Helpers
# ============================================================================

def _max_backtick_run(s: str) -> int:
    """Find the longest consecutive run of backticks in a string."""
    max_run = run = 0
    for ch in s:
        if ch == "`":
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run


def _choose_fence(content: str) -> str:
    """Choose a backtick fence longer than any backtick run in content."""
    return "`" * max(3, _max_backtick_run(content) + 1)


def _language_from_filename(path: str) -> str:
    """Detect language from file extension."""
    name = os.path.basename(path).lower()
    if name == "dockerfile":
        return "dockerfile"
    ext = os.path.splitext(name)[1]
    langs = {
        ".py": "python", ".js": "javascript", ".ts": "typescript",
        ".sh": "bash", ".bash": "bash", ".json": "json",
        ".yaml": "yaml", ".yml": "yaml", ".md": "markdown",
        ".txt": "text", ".go": "go", ".rs": "rust",
        ".java": "java", ".c": "c", ".cpp": "cpp",
        ".toml": "toml", ".ini": "ini",
    }
    return langs.get(ext, "")


def _read_file(path: str) -> str:
    """Read a text file."""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _resolve_path(p: str) -> str:
    """Resolve a path with expansion."""
    p = p.strip().strip("'\"")
    return os.path.abspath(os.path.expandvars(os.path.expanduser(p)))


def _parse_file_commands(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    """Extract /file PATH lines and return (clean_text, attachments)."""
    attachments: List[Tuple[str, str]] = []
    kept_lines: List[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("/file"):
            path = None
            if stripped.startswith("/file:"):
                path = _resolve_path(stripped.split(":", 1)[1])
            else:
                parts = line.split(None, 1)
                if len(parts) == 2:
                    path = _resolve_path(parts[1])

            if path:
                try:
                    content = _read_file(path)
                    attachments.append((path, content))
                    continue
                except OSError as e:
                    kept_lines.append(line)
                    kept_lines.append(f"[Error reading {path}: {e}]")
                    continue

        kept_lines.append(line)

    return "\n".join(kept_lines), attachments


def _build_message(prompt: str, attachments: List[Tuple[str, str]]) -> str:
    """Build final message including file attachments in code blocks."""
    if not attachments:
        return prompt

    parts = [prompt, ""]
    for path, content in attachments:
        fence = _choose_fence(content)
        lang = _language_from_filename(path)
        if not content.endswith("\n"):
            content += "\n"

        if lang:
            block = f"[FILE: {path}]\n{fence}{lang}\n{content}{fence}"
        else:
            block = f"[FILE: {path}]\n{fence}\n{content}{fence}"
        parts.append(block)

    return "\n\n".join(parts)


# ============================================================================
# Interactive TUI
# ============================================================================

def run_interactive(agent, session_id: str = "main_session") -> None:
    """Run interactive TUI with prompt_toolkit."""
    dispatcher = CommandDispatcher(agent)

    # Display banner and help
    display_banner(agent)
    dispatcher._handle_help("", [])

    # Setup completer
    try:
        completer = FuzzyCompleter(NestedCompleter.from_nested_dict(dispatcher.get_completions()))
    except AttributeError:
        completer = FuzzyCompleter(NestedCompleter(dispatcher.get_completions()))

    # Style
    style = Style.from_dict({"prompt": "fg:#888888"})
    prompt_tokens = [("class:prompt", "▌ ")]

    # Key bindings
    kb = KeyBindings()
    menu_visible = Condition(lambda: get_app().current_buffer.complete_state is not None)

    @kb.add("enter", filter=menu_visible)
    def _(event):
        """Apply completion on Enter when menu is visible."""
        b = event.current_buffer
        if b.complete_state:
            if b.complete_state.current_completion is None:
                b.complete_next()
            if b.complete_state and b.complete_state.current_completion:
                b.apply_completion(b.complete_state.current_completion)

    @kb.add("enter", filter=~menu_visible)
    def _(event):
        """Submit on Enter when menu is not visible."""
        if event.app.current_buffer.document.text.strip():
            event.app.current_buffer.validate_and_handle()

    @kb.add("down", filter=menu_visible)
    def _(event):
        event.current_buffer.complete_next()

    @kb.add("up", filter=menu_visible)
    def _(event):
        event.current_buffer.complete_previous()

    @kb.add("tab")
    def _(event):
        event.current_buffer.complete_next()

    @kb.add("s-tab")
    def _(event):
        event.current_buffer.complete_previous()

    # Shift/Ctrl/Alt-Enter for newlines
    for key in ("s-enter", "c-enter", "a-enter", "c-j"):
        try:
            @kb.add(key)
            def _(event):
                event.current_buffer.insert_text("\n")
        except Exception:
            pass

    @kb.add("c-s", filter=~menu_visible)
    def _(event):
        event.app.current_buffer.validate_and_handle()

    @kb.add("c-q")
    def _(event):
        event.app.exit(exception=EOFError)

    @kb.add("c-c")
    def _(event):
        event.app.exit(exception=KeyboardInterrupt)

    session = PromptSession(
        style=style,
        completer=completer,
        key_bindings=kb,
    )

    # REPL loop
    try:
        while True:
            with patch_stdout():
                try:
                    text = session.prompt(
                        prompt_tokens,
                        multiline=True,
                        prompt_continuation=lambda w, l, s: prompt_tokens,
                        complete_while_typing=True,
                        reserve_space_for_menu=8,
                    )
                except EOFError:
                    break

            if not text:
                continue

            # Dispatch command
            result = dispatcher.dispatch(text)

            if result.should_exit:
                break

            if result.handled and not result.should_process:
                continue

            # Send to agent
            if result.should_process:
                width = shutil.get_terminal_size(fallback=(80, 20)).columns
                console.print("[dim]" + "-" * min(width, 120) + "[/dim]")

                clean, attachments = _parse_file_commands(text)
                message = _build_message(clean, attachments)
                stream_response(agent, message, session_id)

                console.print("[dim]" + "-" * min(width, 120) + "[/dim]")

    except KeyboardInterrupt:
        pass


# ============================================================================
# Entry Points
# ============================================================================

def run_once(agent, prompt: str, session_id: str = "main_session") -> None:
    """Run a single query and exit."""
    stream_response(agent, prompt, session_id)


def run(agent, args: List[str] = None) -> None:
    """Main entry point for CLI."""
    if args is None:
        args = sys.argv[1:]

    try:
        if args:
            run_once(agent, " ".join(args))
        else:
            run_interactive(agent)
    except KeyboardInterrupt:
        sys.exit(130)
