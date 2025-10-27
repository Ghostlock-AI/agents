"""
Base Reasoning Strategy - Abstract interface for all reasoning strategies.

All reasoning strategies must implement this interface to be compatible with the agent system.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class ReasoningStrategy(ABC):
    """Abstract base class for reasoning strategies."""

    @abstractmethod
    def create_graph(self, agent_state_class, llm_with_tools, tools):
        """
        Create and return a compiled LangGraph for this reasoning strategy.

        Args:
            agent_state_class: TypedDict defining the state schema
            llm_with_tools: LLM with tools bound via llm.bind_tools()
            tools: List of tool instances

        Returns:
            Compiled LangGraph ready to execute
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Get the unique identifier for this strategy.

        Returns:
            Strategy name (lowercase, no spaces, e.g., 'react', 'rewoo')
        """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """
        Get a human-readable description of this strategy.

        Returns:
            Description explaining when and why to use this strategy
        """
        pass

    def get_config(self) -> Dict[str, Any]:
        """
        Get strategy-specific configuration parameters.

        Returns:
            Dict of configuration options and their current values
        """
        return {}

    def update_config(self, **kwargs) -> None:
        """
        Update strategy-specific configuration.

        Args:
            **kwargs: Configuration parameters to update
        """
        pass

    def supports_streaming(self) -> bool:
        """
        Check if this strategy supports token streaming.

        Returns:
            True if streaming is supported, False otherwise
        """
        return True

    def get_trace_info(self, state: Optional[Any] = None) -> Dict[str, Any]:
        """
        Get debug/trace information about the current reasoning step.

        Args:
            state: Current graph state (optional)

        Returns:
            Dict with tracing information (steps taken, current node, etc.)
        """
        return {
            "strategy": self.get_name(),
            "description": self.get_description(),
        }
