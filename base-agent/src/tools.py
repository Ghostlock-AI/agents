"""
Tools - Tool definitions for the agent.

Tool descriptions are CRITICAL - they tell the LLM when and how to use each tool.
LangChain/LangGraph uses these descriptions to bind tools to the model automatically.
"""

import subprocess
import requests
from typing import Optional
from ddgs import DDGS
from langchain_core.tools import tool


def get_tools():
    """
    Get all tools for the agent.
    Must be called after environment variables are loaded.
    """
    return [ddgs_search, web_fetch, shell_exec]


@tool
def ddgs_search(query: str, max_results: int = 5, return_json: bool = False) -> str:
    """Search the web using DuckDuckGo for current information, documentation, or answers to questions.

    WHEN TO USE:
    - Finding current events, news, or recent information
    - Looking up documentation or technical resources
    - Researching topics not in training data (post-2023)
    - Verifying facts or gathering multiple perspectives

    BEST PRACTICES:
    - Use specific, focused queries (e.g., "Python asyncio tutorial 2024")
    - Follow up with web_fetch to get full content from top results
    - For research tasks, use max_results=3-5 then fetch top URLs

    Args:
        query: Specific search query (e.g., "latest Python release notes")
        max_results: Number of results to return (default: 5, max: 10)
        return_json: If True, returns structured JSON for programmatic parsing

    Returns:
        Formatted search results with titles, URLs, and snippets

    Examples:
        ddgs_search("React hooks documentation")
        ddgs_search("weather San Francisco today", max_results=3)
        ddgs_search("AI news 2024", return_json=True)

    AVOID:
    - Vague queries like "stuff" or "things"
    - Queries better answered with date/time commands
    - Searching for information you already have
    """
    try:
        ddgs = DDGS()
        results = ddgs.text(query, max_results=max_results)

        if not results:
            return "No results found for your search query."

        # Optionally return JSON for deterministic post-processing
        if return_json:
            import json as _json
            items = [
                {"title": r.get("title"), "url": r.get("href"), "snippet": r.get("body")}
                for r in results
            ]
            return _json.dumps(items, ensure_ascii=False)

        # Text format with easy-to-parse URL lines
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append(
                f"{i}. {result['title']}\n"
                f"   URL: {result['href']}\n"
                f"   {result['body']}"
            )

        return "\n\n".join(formatted_results)

    except Exception as e:
        return f"Error performing search: {str(e)}"


@tool
def web_fetch(url: str, timeout: int = 10) -> str:
    """Fetch and extract text content from a specific webpage.

    WHEN TO USE:
    - After ddgs_search to get full content from promising URLs
    - Reading documentation, articles, or blog posts
    - Extracting detailed information from a known URL
    - Verifying information from primary sources

    BEST PRACTICES:
    - Use after search to fetch 1-3 most relevant URLs
    - Works best with text-heavy pages (docs, articles, blogs)
    - Automatically handles http/https schemes
    - Content is truncated at 5000 chars for manageability

    Args:
        url: Full URL or domain (e.g., "https://example.com/page" or "example.com/page")
        timeout: Max seconds to wait for response (default: 10)

    Returns:
        Extracted text content (HTML tags removed, whitespace normalized)

    Examples:
        web_fetch("https://docs.python.org/3/library/asyncio.html")
        web_fetch("github.com/user/repo/blob/main/README.md")
        web_fetch("example.com/article", timeout=15)

    LIMITATIONS:
    - JavaScript-rendered content may not be captured
    - Images, videos not extracted (text only)
    - Very long pages truncated at 5000 characters
    - Some sites may block automated requests

    AVOID:
    - Fetching the same URL multiple times
    - Non-text content (PDFs, images, videos)
    - Sites requiring authentication
    """
    try:
        # Add scheme if missing
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }

        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()

        # Simple text extraction - removes most HTML tags
        content = response.text

        # Basic HTML to text conversion
        import re

        # Remove script and style tags
        content = re.sub(r"<script[^>]*>.*?</script>", "", content, flags=re.DOTALL)
        content = re.sub(r"<style[^>]*>.*?</style>", "", content, flags=re.DOTALL)

        # Remove HTML tags
        content = re.sub(r"<[^>]+>", " ", content)

        # Clean up whitespace
        content = re.sub(r"\s+", " ", content)
        content = content.strip()

        if len(content) > 5000:
            content = content[:5000] + "...\n[Content truncated]"

        return content if content else "Could not extract readable content from the webpage."

    except requests.RequestException as e:
        return f"Error fetching URL: {str(e)}"
    except Exception as e:
        return f"Error processing webpage: {str(e)}"


# Shell Execution Tool
# Custom tool using @tool decorator - description is in the docstring
@tool
def shell_exec(command: str) -> str:
    """Execute Unix shell commands for file operations, git, system info, and text processing.

    WHEN TO USE:
    - Getting current date/time (date command)
    - Listing files and directories (ls, find)
    - Viewing file contents (cat, head, tail)
    - Git operations (git status, git log, git diff)
    - System information (whoami, pwd, uname)
    - Text processing (grep, wc, sort)

    BEST PRACTICES:
    - For date/time: use `date` command with specific format
    - Use absolute paths when possible
    - Combine commands with && for sequential execution
    - Use quotes for arguments with spaces

    Args:
        command: Shell command to execute (e.g., "ls -la /tmp")

    Returns:
        Command output (stdout and stderr combined)

    Examples:
        shell_exec("date '+%Y-%m-%d %H:%M:%S'")     # Current date and time
        shell_exec("ls -la ~/Documents")             # List files with details
        shell_exec("git log --oneline -5")          # Last 5 commits
        shell_exec("cat README.md | head -20")      # First 20 lines of file
        shell_exec("find . -name '*.py' -type f")   # Find Python files

    SAFETY LIMITS:
    - Commands timeout after 30 seconds
    - Runs in current working directory
    - Has access to file system (use with caution)

    STRICTLY AVOID:
    - Destructive operations: rm -rf, dd, mkfs
    - System modifications: sudo, chmod 777, chown
    - Interactive commands: vim, nano, top
    - Network attacks: nmap, nc, curl loops
    - Resource exhaustion: fork bombs, infinite loops
    - Package installation: apt, yum, pip install

    SAFE COMMANDS:
    ✅ date, ls, pwd, whoami, cat, head, tail, grep, find, wc
    ✅ git status, git log, git diff, git branch
    ✅ echo, printf, basename, dirname
    ❌ rm, sudo, chmod, chown, kill, pkill, dd, mkfs
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,  # 30 second timeout
        )

        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"

        return output if output else "Command executed successfully (no output)"

    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds"
    except Exception as e:
        return f"Error executing command: {str(e)}"
