"""
Tools - Tool definitions for the agent.

Tool descriptions are CRITICAL - they tell the LLM when and how to use each tool.
LangChain/LangGraph uses these descriptions to bind tools to the model automatically.
"""

import os
import subprocess

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool


def get_tools():
    """
    Get all tools for the agent.
    Must be called after environment variables are loaded.
    """
    return [get_tavily_search(), shell_exec]


def get_tavily_search():
    """
    Initialize Tavily search tool.
    Called after dotenv loads environment variables.
    """
    tavily = TavilySearchResults(
        max_results=5,
        search_depth="advanced",
        include_answer=True,
        include_raw_content=False,
        api_key=os.getenv("TAVILY_API_KEY"),
    )

    # Override description to be more specific for our use case
    tavily.description = (
        "Search the web for current information, documentation, or answers to questions. "
        "Use this when you need up-to-date information that isn't in your training data, "
        "or when the user explicitly asks you to search for something. "
        "Input should be a search query string."
    )

    return tavily


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
