"""
Shared tool context and examples used across reasoning strategies.

Provides a consistent, centralized block describing available tools and
how to call them explicitly via tool_calls. Strategies can include this
block in their prompts to ensure uniform behavior regardless of which
reasoning graph is active.
"""

from typing import List


def build_tool_guide(tools: List) -> str:
    """Return a consistent tool catalog + usage examples string.

    Includes:
    - Tool catalog (name + description)
    - Rules for explicit tool_calls
    - Examples for ddgs_search (JSON and text) and web_fetch, shell_exec
    """
    lines = []
    lines.append("TOOL CATALOG:")
    for t in tools:
        name = getattr(t, "name", "tool")
        desc = getattr(t, "description", "")
        lines.append(f"- {name}: {desc}")
    lines.append("")
    lines.append("RULES:")
    lines.append("- Emit explicit tool_calls with fields: name, args, id (unique).")
    lines.append("- If a message contains tool_calls, the next step must execute them.")
    lines.append("- For research: perform ddgs_search → web_fetch (top 1–2 links) before answering.")
    lines.append("- If ddgs_search returns text, parse lines starting with 'URL:' to get links.")
    lines.append("- If ddgs_search returns JSON, use the 'url' field from the objects.")
    lines.append("")
    lines.append("EXAMPLES (tool_calls objects):")
    lines.append("- ddgs_search (JSON output):")
    lines.append("  {\"name\": \"ddgs_search\", \"args\": {\"query\": \"weather San Francisco today date\", \"max_results\": 5, \"return_json\": true}, \"id\": \"s1\"}")
    lines.append("- ddgs_search (text output):")
    lines.append("  {\"name\": \"ddgs_search\", \"args\": {\"query\": \"weather San Francisco today date\", \"max_results\": 5}, \"id\": \"s1\"}")
    lines.append("- web_fetch:")
    lines.append("  {\"name\": \"web_fetch\", \"args\": {\"url\": \"https://example.com/article\"}, \"id\": \"s2\"}")
    lines.append("- shell_exec:")
    lines.append("  {\"name\": \"shell_exec\", \"args\": {\"command\": \"ls -la\"}, \"id\": \"s3\"}")
    return "\n".join(lines)

