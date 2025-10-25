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

    Use this when you need up-to-date information that isn't in your training data,
    or when the user explicitly asks you to search for something.

    Args:
        query: The search query string
        max_results: Maximum number of results to return (default: 5)
        return_json: If True, return a compact JSON array of {title, url, snippet}.

    Returns:
        Formatted search results with titles, snippets, and links
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
    """Fetch and extract the text content from a specific webpage.

    Use this to retrieve the full content of a URL for detailed information.
    Works best for text-based content (docs, articles, etc).

    Args:
        url: The URL to fetch
        timeout: Request timeout in seconds (default: 10)

    Returns:
        The extracted text content from the webpage
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

    Use for: listing files, viewing contents, searching, git operations, running programs.

    Args:
        command: Shell command to execute

    Returns:
        Command output

    Note: Commands timeout after 30s. Avoid destructive operations (rm -rf, dd, sudo).
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
