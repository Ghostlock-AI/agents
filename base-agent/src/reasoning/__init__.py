"""
Reasoning module - Workflows and Agents for intelligent task execution.

This module distinguishes between two fundamental patterns:

WORKFLOWS: Predefined code paths with LLM and tool orchestration
- ReWOO: Plans all steps upfront, executes in parallel
- Plan-Execute: Adaptive planning with replanning capability

AGENTS: LLMs dynamically control their own processes and tool usage
- ReAct: Iterative reasoning with tool use
- LATS: Tree search with self-reflection

Usage:
    from reasoning import get_global_registry

    registry = get_global_registry()
    strategy = registry.get_current_strategy()
    graph = strategy.create_graph(state_class, llm, tools)
"""

from .strategy_registry import (
    StrategyRegistry,
    get_global_registry,
    reset_global_registry,
)

from .base import ReasoningStrategy

# Import agents
from .agents import (
    ReActStrategy,
    LATSStrategy,
)

# Import workflows
from .workflows import (
    ReWOOStrategy,
    PlanExecuteStrategy,
)

# Legacy support - keep the old create_react_graph function for backward compatibility
def create_react_graph(agent_state_class, llm_with_tools, tools):
    """
    Create a ReAct reasoning graph (legacy function).

    For new code, prefer using the strategy registry:
        registry = get_global_registry()
        strategy = registry.get_current_strategy()
        graph = strategy.create_graph(state_class, llm, tools)
    """
    strategy = ReActStrategy()
    return strategy.create_graph(agent_state_class, llm_with_tools, tools)


__all__ = [
    # Registry
    "StrategyRegistry",
    "get_global_registry",
    "reset_global_registry",

    # Strategies
    "ReasoningStrategy",
    "ReActStrategy",
    "ReWOOStrategy",
    "PlanExecuteStrategy",
    "LATSStrategy",

    # Legacy
    "create_react_graph",
]
