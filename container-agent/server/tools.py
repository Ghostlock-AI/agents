"""
LangChain tools for search, web fetch, and shell execution.
"""

import os
import warnings
from typing import List

import requests

# Suppress all warnings
warnings.filterwarnings("ignore")

from langchain_community.tools import ShellTool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool


@tool("web_fetch")
def web_fetch(url: str) -> str:
    """Fetch and return the raw text content of a web resource.

    Use this tool to read pages discovered via search or to interact with HTTP
    services directly when investigating or gathering evidence. Accepts http/https
    URLs and returns the response body text (truncated for brevity).
    """
    try:
        if not (url.startswith("http://") or url.startswith("https://")):
            return "Error: URL must start with http:// or https://"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        text = resp.text or ""
        # Truncate to 20000 chars to ensure long payloads are visible
        return text[:20000]
    except Exception as e:  # noqa: BLE001 (surface tool error textually)
        return f"Error fetching URL: {e}"


def _maybe_make_search_tool():
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        # No exit: degrade gracefully so the server can still run
        return None
    try:
        return TavilySearchResults(
            max_results=5,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=False,
            include_images=False,
        )
    except Exception:
        # If key is invalid, return None and let the agent proceed without search
        return None


def get_tools() -> List:
    """Return available tools based on environment.

    Includes ShellTool and web_fetch by default; adds Tavily search if configured.
    """
    tools: List = []
    search_tool = _maybe_make_search_tool()
    if search_tool is not None:
        tools.append(search_tool)
    tools.append(web_fetch)
    tools.append(ShellTool())
    return tools

