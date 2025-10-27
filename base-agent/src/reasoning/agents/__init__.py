"""
Agent Patterns

Agents allow LLMs to dynamically control their own processes and tool usage.
They adapt based on environmental feedback and are suited for open-ended
problems where step sequences cannot be predetermined.

Available Agents:
- ReAct: Iterative reasoning with tool use (Reason → Act → Observe loop)
- LATS: Language Agent Tree Search with self-reflection

When to use agents:
- Open-ended problems where steps cannot be predetermined
- Tasks requiring dynamic adaptation based on results
- Problems where exploration and trial-and-error are valuable
- Situations where you trust the LLM to make good decisions
"""

from reasoning.agents.react import ReActStrategy
from reasoning.agents.lats import LATSStrategy

__all__ = ["ReActStrategy", "LATSStrategy"]
