"""
Reasoning module - Advanced reasoning strategies for the agent.

This module provides multiple reasoning strategies:
- ReAct: Iterative reasoning with tool use
- ReWOO: Plan all steps upfront, execute in parallel
- Plan-and-Execute: Adaptive planning with sequential execution
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

from .strategies import (
    ReasoningStrategy,
    ReActStrategy,
    ReWOOStrategy,
    PlanExecuteStrategy,
    LATSStrategy,
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
