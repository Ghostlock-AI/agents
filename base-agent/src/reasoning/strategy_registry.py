"""
Reasoning Strategy Registry

Manages available reasoning strategies and allows runtime switching.
Provides a centralized place to register and retrieve strategies.
"""

from typing import Dict, List, Optional
from .base import ReasoningStrategy
from .agents import ReActStrategy, LATSStrategy
from .workflows import (
    PromptChainWorkflow,
    RoutingWorkflow,
    ReWOOStrategy,
    PlanExecuteStrategy,
)


class StrategyRegistry:
    """Registry for reasoning strategies."""

    def __init__(self):
        """Initialize the registry with default strategies."""
        self._strategies: Dict[str, ReasoningStrategy] = {}
        self._current_strategy_name: str = "react"

        # Register default strategies
        self._register_defaults()

    def _register_defaults(self):
        """Register the default built-in strategies."""
        # Agents: Dynamic LLM control
        agents = [
            ReActStrategy(),
            LATSStrategy(),
        ]

        # Workflows: Predefined code paths
        workflows = [
            PromptChainWorkflow(),  # Generate → Validate → Refine
            RoutingWorkflow(),      # Classify → Route to handler
            ReWOOStrategy(),        # Plan → Execute → Synthesize
            PlanExecuteStrategy(),  # Plan → Execute → Replan if needed
        ]

        # Register all strategies
        for strategy in agents + workflows:
            self.register(strategy)

    def register(self, strategy: ReasoningStrategy) -> None:
        """
        Register a new reasoning strategy.

        Args:
            strategy: The strategy instance to register

        Raises:
            ValueError: If strategy name conflicts with existing strategy
        """
        name = strategy.get_name()

        if name in self._strategies:
            raise ValueError(
                f"Strategy '{name}' is already registered. "
                f"Use a different name or unregister the existing strategy first."
            )

        self._strategies[name] = strategy

    def unregister(self, name: str) -> None:
        """
        Unregister a strategy.

        Args:
            name: Name of the strategy to remove

        Raises:
            KeyError: If strategy doesn't exist
        """
        if name not in self._strategies:
            raise KeyError(f"Strategy '{name}' is not registered")

        if name == self._current_strategy_name:
            raise ValueError(
                f"Cannot unregister '{name}' because it's the current strategy. "
                f"Switch to a different strategy first."
            )

        del self._strategies[name]

    def get_strategy(self, name: Optional[str] = None) -> ReasoningStrategy:
        """
        Get a strategy by name.

        Args:
            name: Name of the strategy (defaults to current strategy)

        Returns:
            The requested strategy instance

        Raises:
            KeyError: If strategy doesn't exist
        """
        strategy_name = name or self._current_strategy_name

        if strategy_name not in self._strategies:
            available = ", ".join(self._strategies.keys())
            raise KeyError(
                f"Strategy '{strategy_name}' not found. "
                f"Available strategies: {available}"
            )

        return self._strategies[strategy_name]

    def get_current_strategy(self) -> ReasoningStrategy:
        """Get the currently active strategy."""
        return self.get_strategy(self._current_strategy_name)

    def set_current_strategy(self, name: str) -> None:
        """
        Set the current active strategy.

        Args:
            name: Name of the strategy to activate

        Raises:
            KeyError: If strategy doesn't exist
        """
        if name not in self._strategies:
            available = ", ".join(self._strategies.keys())
            raise KeyError(
                f"Strategy '{name}' not found. "
                f"Available strategies: {available}"
            )

        self._current_strategy_name = name

    def get_current_strategy_name(self) -> str:
        """Get the name of the currently active strategy."""
        return self._current_strategy_name

    def list_strategies(self) -> List[Dict[str, str]]:
        """
        List all registered strategies with their descriptions.

        Returns:
            List of dicts with 'name', 'description', and 'is_current' keys
        """
        return [
            {
                "name": name,
                "description": strategy.get_description(),
                "is_current": name == self._current_strategy_name,
            }
            for name, strategy in self._strategies.items()
        ]

    def get_strategy_info(self, name: str) -> Dict:
        """
        Get detailed information about a specific strategy.

        Args:
            name: Name of the strategy

        Returns:
            Dict with strategy details including config

        Raises:
            KeyError: If strategy doesn't exist
        """
        strategy = self.get_strategy(name)

        return {
            "name": strategy.get_name(),
            "description": strategy.get_description(),
            "config": strategy.get_config(),
            "supports_streaming": strategy.supports_streaming(),
            "is_current": name == self._current_strategy_name,
        }

    def update_strategy_config(self, name: str, **kwargs) -> None:
        """
        Update configuration for a specific strategy.

        Args:
            name: Name of the strategy
            **kwargs: Configuration parameters to update

        Raises:
            KeyError: If strategy doesn't exist
        """
        strategy = self.get_strategy(name)
        strategy.update_config(**kwargs)


# Global registry instance
_global_registry: Optional[StrategyRegistry] = None


def get_global_registry() -> StrategyRegistry:
    """
    Get the global strategy registry (singleton pattern).

    Returns:
        The global StrategyRegistry instance
    """
    global _global_registry

    if _global_registry is None:
        _global_registry = StrategyRegistry()

    return _global_registry


def reset_global_registry() -> None:
    """Reset the global registry (useful for testing)."""
    global _global_registry
    _global_registry = None
