"""
Todo management tools: track multi-step tasks

Enables agents to organize complex workflows and track progress.
"""

from langchain_core.tools import tool
from typing import List, Dict, Optional
from pathlib import Path
import json
from tool_logger import log_tool_start, log_tool_complete, log_tool_error


# In-memory task storage (will be session-specific)
_task_store: Dict[str, List[Dict[str, str]]] = {}


@tool
def todo_write(tasks: str, session_id: str = "default") -> str:
    """Create or update task list for tracking multi-step operations.

    Args:
        tasks: JSON array string of task objects with {content, status, activeForm}
               status must be: 'pending' | 'in_progress' | 'completed'
        session_id: Session identifier (default: "default")

    Returns:
        Summary of task list state

    Examples:
        todo_write('[{"content": "Compile program", "status": "completed", "activeForm": "Compiling program"}]')
        todo_write('[{"content": "Run tests", "status": "in_progress", "activeForm": "Running tests"}]')

    Task format:
        {
            "content": "Do something",           # Imperative form
            "status": "pending",                   # pending | in_progress | completed
            "activeForm": "Doing something"        # Present continuous form
        }
    """
    log_tool_start("TodoWrite", args_str=f"session={session_id}")

    try:
        # Parse JSON task list
        task_list = json.loads(tasks)

        if not isinstance(task_list, list):
            log_tool_error("tasks must be a JSON array")
            return "ERROR: tasks parameter must be a JSON array of task objects"

        # Validate task structure
        for i, task in enumerate(task_list):
            if not isinstance(task, dict):
                log_tool_error(f"Task {i} is not an object")
                return f"ERROR: Task {i} must be an object, got {type(task)}"

            required_fields = ["content", "status", "activeForm"]
            for field in required_fields:
                if field not in task:
                    log_tool_error(f"Task {i} missing '{field}'")
                    return f"ERROR: Task {i} missing required field '{field}'"

            if task["status"] not in ["pending", "in_progress", "completed"]:
                log_tool_error(f"Invalid status: {task['status']}")
                return f"ERROR: status must be 'pending', 'in_progress', or 'completed', got '{task['status']}'"

        # Store tasks
        _task_store[session_id] = task_list

        # Generate summary
        total = len(task_list)
        completed = sum(1 for t in task_list if t["status"] == "completed")
        in_progress = sum(1 for t in task_list if t["status"] == "in_progress")
        pending = sum(1 for t in task_list if t["status"] == "pending")

        summary = f"Task list updated: {total} tasks ({completed} completed, {in_progress} in progress, {pending} pending)"
        log_tool_complete(summary)

        return f"""Task List Updated (session: {session_id})

Total: {total} tasks
  ✓ Completed: {completed}
  → In Progress: {in_progress}
  ○ Pending: {pending}

Current tasks:
{_format_task_list(task_list)}
"""

    except json.JSONDecodeError as e:
        log_tool_error(f"Invalid JSON: {e}")
        return f"ERROR: Invalid JSON in tasks parameter: {e}"
    except Exception as e:
        log_tool_error(str(e))
        return f"ERROR: {e}"


@tool
def todo_read(session_id: str = "default") -> str:
    """Read current task list state.

    Args:
        session_id: Session identifier (default: "default")

    Returns:
        Formatted task list with status indicators

    Example:
        todo_read() - get current task list
        todo_read("custom_session") - get specific session's tasks
    """
    log_tool_start("TodoRead", args_str=f"session={session_id}")

    try:
        if session_id not in _task_store or not _task_store[session_id]:
            log_tool_complete("No tasks found")
            return f"No tasks found for session: {session_id}"

        task_list = _task_store[session_id]
        total = len(task_list)
        completed = sum(1 for t in task_list if t["status"] == "completed")
        in_progress = sum(1 for t in task_list if t["status"] == "in_progress")
        pending = sum(1 for t in task_list if t["status"] == "pending")

        log_tool_complete(f"{total} tasks ({completed} completed)")

        return f"""Task List (session: {session_id})

Total: {total} tasks
  ✓ Completed: {completed}
  → In Progress: {in_progress}
  ○ Pending: {pending}

Tasks:
{_format_task_list(task_list)}

Progress: {completed}/{total} completed ({int(completed/total*100) if total > 0 else 0}%)
"""

    except Exception as e:
        log_tool_error(str(e))
        return f"ERROR: {e}"


def _format_task_list(tasks: List[Dict[str, str]]) -> str:
    """Format task list for display."""
    lines = []
    for i, task in enumerate(tasks, 1):
        status = task["status"]
        content = task["content"]

        if status == "completed":
            icon = "✓"
            status_display = "completed"
        elif status == "in_progress":
            icon = "→"
            status_display = "in_progress"
        else:
            icon = "○"
            status_display = "pending"

        lines.append(f"  {i}. {icon} [{status_display}] {content}")

    return "\n".join(lines)
