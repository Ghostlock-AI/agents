"""
Agent - Core agent class with reasoning loop and memory.
"""

import os
from typing import TypedDict, Annotated, Optional, Dict, Any, List
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages

from tools import get_tools
from reasoning.tool_context import build_tool_guide
from reasoning import get_global_registry, create_react_graph


class AgentState(TypedDict):
    """State that tracks conversation messages and optional planning state."""
    messages: Annotated[list[BaseMessage], add_messages]
    # Optional planning state used by some strategies (ReWOO / Plan-Execute)
    plan: Optional[Dict[str, Any]]  # Strategy-specific plan structure
    step_idx: Optional[int]         # Current step index for sequential execution
    executed: Optional[List[str]]   # Executed step ids (as strings)
    results: Optional[List[Dict[str, Any]]]  # Collected tool results / summaries


class Agent:
    """Base agent with LLM, reasoning strategies, tools, and memory."""

    def __init__(
        self,
        model_name: str = None,
        temperature: float = 0.7,
        system_prompt_path: str = "system_prompt.txt",
        tools: list = None,
        reasoning_strategy: str = "react"
    ):
        """Initialize the agent."""
        self.model_name = model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = temperature
        self.system_prompt = self._load_system_prompt(system_prompt_path)

        # Initialize tools after env is loaded
        self.tools = tools or get_tools()

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            streaming=True
        )

        # Bind tools to LLM
        # This tells the LLM about available tools and their descriptions
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Initialize strategy registry
        self.strategy_registry = get_global_registry()

        # Set initial reasoning strategy
        if reasoning_strategy:
            self.strategy_registry.set_current_strategy(reasoning_strategy)

        # Build the reasoning graph using current strategy
        self.graph = self._build_graph()

    def _load_system_prompt(self, path: str) -> str:
        """Load system prompt from file."""
        prompt_path = Path(__file__).parent / path

        if prompt_path.exists():
            return prompt_path.read_text().strip()

        return "You are a helpful AI assistant."

    def _build_graph(self):
        """Build the reasoning graph using the current strategy."""
        strategy = self.strategy_registry.get_current_strategy()
        return strategy.create_graph(AgentState, self.llm_with_tools, self.tools)

    def switch_reasoning_strategy(self, strategy_name: str):
        """
        Switch to a different reasoning strategy.

        Args:
            strategy_name: Name of the strategy to switch to (e.g., 'react', 'rewoo', 'plan-execute', 'lats')

        Raises:
            KeyError: If strategy doesn't exist
        """
        self.strategy_registry.set_current_strategy(strategy_name)
        # Rebuild the graph with the new strategy
        self.graph = self._build_graph()

    def get_current_strategy_name(self) -> str:
        """Get the name of the currently active reasoning strategy."""
        return self.strategy_registry.get_current_strategy_name()

    def list_strategies(self) -> list:
        """List all available reasoning strategies with descriptions."""
        return self.strategy_registry.list_strategies()

    def get_strategy_info(self, strategy_name: str = None) -> dict:
        """
        Get detailed information about a strategy.

        Args:
            strategy_name: Name of the strategy (defaults to current strategy)

        Returns:
            Dict with strategy details including config
        """
        if strategy_name is None:
            strategy_name = self.get_current_strategy_name()
        return self.strategy_registry.get_strategy_info(strategy_name)

    def stream(self, user_input: str, thread_id: str = "default", show_trace: bool = False):
        """
        Stream response for user input.

        Args:
            user_input: The user's query
            thread_id: Session thread identifier
            show_trace: Whether to show reasoning trace information
        """
        from langchain_core.messages import ToolMessage

        # Prepend system message to the user input
        input_state = {
            "messages": [
                SystemMessage(content=self.system_prompt),
                # Inject shared tool guide so all strategies see the same context
                SystemMessage(content=build_tool_guide(self.tools)),
                HumanMessage(content=user_input),
            ]
        }

        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }

        # Get strategy trace info if requested
        if show_trace:
            strategy = self.strategy_registry.get_current_strategy()
            trace_info = strategy.get_trace_info()
            yield f"[TRACE: {trace_info.get('strategy', 'unknown').upper()}]\n"

        # Stream and yield AI message content
        for chunk in self.graph.stream(input_state, config, stream_mode="values"):
            if "messages" in chunk and chunk["messages"]:
                last_message = chunk["messages"][-1]

                # Check for tool calls in AI messages
                if isinstance(last_message, AIMessage):
                    # Tool calls are already logged by tool_logger, no need to show [TOOL: xxx]

                    # Show strategy-specific markers for LATS and Plan-Execute
                    if last_message.content:
                        content = last_message.content
                        # Show trace markers for special reasoning steps
                        if "[CANDIDATES GENERATED]" in content or "[PLAN CREATED]" in content or "[REFLECTION]" in content:
                            yield content
                        # Only yield if there's actual content (not just tool calls)
                        elif content:
                            yield content

                # Show tool results
                elif isinstance(last_message, ToolMessage):
                    # Don't display tool output, just let it feed back to agent
                    pass
