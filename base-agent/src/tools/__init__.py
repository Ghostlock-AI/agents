"""
Tools - Tool definitions for the agent.

Tool descriptions are CRITICAL - they tell the LLM when and how to use each tool.
LangChain/LangGraph uses these descriptions to bind tools to the model automatically.
"""

import requests
from typing import Optional
from ddgs import DDGS
from langchain_core.tools import tool
from tool_logger import log_tool_start, log_tool_complete, log_tool_error, format_bytes

# Import specialized tools
from tools.file_ops import read_file, write_file, edit_file
from tools.search import grep_code, find_files
from tools.execution import bash_exec
from tools.directory import list_directory, tree_view
from tools.patch import apply_patch, diff_files
from tools.todo import todo_write, todo_read

# Import memory tools
from memories import save_persistent_memory, search_persistent_memories


def get_tools():
    """
    Get all tools for the agent.
    Must be called after environment variables are loaded.

    Tool organization:
    - File operations: read_file, write_file, edit_file
    - Directory: list_directory, tree_view
    - Search: grep_code, find_files
    - Patch: apply_patch, diff_files
    - Execution: bash_exec
    - Web: ddgs_search, web_fetch
    - Task management: todo_write, todo_read
    - Memory: save_persistent_memory, search_persistent_memories
    """
    return [
        # File operations (replaces generic shell for file work)
        read_file,
        write_file,
        edit_file,

        # Directory operations
        list_directory,
        tree_view,

        # Search (replaces grep/find shell commands)
        grep_code,
        find_files,

        # Patch/diff operations
        apply_patch,
        diff_files,

        # Execution (keep for git, compilation, running programs)
        bash_exec,

        # Task management
        todo_write,
        todo_read,

        # Web research
        ddgs_search,
        web_fetch,

        # Memory
        save_persistent_memory,
        search_persistent_memories,
    ]


@tool
def ddgs_search(query: str, max_results: int = 5, return_json: bool = False) -> str:
    """Search the web using DuckDuckGo for current information, documentation, or answers to questions.

    Use this when you need up-to-date information that isn't in your training data,
    or when the user explicitly asks you to search for something.

    Args:
        query: The search query string
        max_results: Maximum number of results to return (default: 5)
        return_json: If True, return a compact JSON array of {title, url, snippet}.

    Returns:
        Formatted search results with titles, snippets, and links
    """
    log_tool_start("WebSearch", args_str=f'"{query}"')

    try:
        ddgs = DDGS()
        results = ddgs.text(query, max_results=max_results)

        if not results:
            log_tool_complete("No results found")
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

        log_tool_complete(f"Found {len(results)} results")
        return "\n\n".join(formatted_results)

    except Exception as e:
        log_tool_error(str(e))
        return f"Error performing search: {str(e)}"


@tool
def web_fetch(url: str, timeout: int = 10) -> str:
    """Fetch and extract the text content from a specific webpage.

    Use this to retrieve the full content of a URL for detailed information.
    Works best for text-based content (docs, articles, etc).

    Args:
        url: The URL to fetch
        timeout: Request timeout in seconds (default: 10)

    Returns:
        The extracted text content from the webpage
    """
    log_tool_start("WebFetch", args_str=url)

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

        # Log completion with size info
        size_kb = len(response.content) / 1024
        log_tool_complete(f"Received {format_bytes(len(response.content))} ({response.status_code} OK)")

        return content if content else "Could not extract readable content from the webpage."

    except requests.RequestException as e:
        log_tool_error(str(e))
        return f"Error fetching URL: {str(e)}"
    except Exception as e:
        log_tool_error(str(e))
        return f"Error processing webpage: {str(e)}"
