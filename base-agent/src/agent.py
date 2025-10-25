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
from context import FileContextManager
from memory import VectorStore


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
        reasoning_strategy: str = "react",
        enable_context: bool = True,
        enable_vector_store: bool = True,
        max_context_tokens: int = 100000
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

        # Initialize context management
        self.context_manager = None
        self.vector_store = None

        if enable_context:
            self.context_manager = FileContextManager(max_tokens=max_context_tokens)
            # Pass LLM to context manager for intelligent summarization
            self.context_manager._llm = self.llm

        if enable_vector_store:
            self.vector_store = VectorStore()

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

    # Context Management Methods

    def add_context_file(self, filepath: str, content: Optional[str] = None) -> tuple[bool, str]:
        """
        Add a file to context (and vector store if enabled).

        Args:
            filepath: Path to the file
            content: Optional file content (will read from disk if not provided)

        Returns:
            Tuple of (success, message)
        """
        if not self.context_manager:
            return False, "Context management is not enabled"

        # Add to context manager
        success, message = self.context_manager.add_file(filepath, content)

        if success and self.vector_store:
            # Also add to vector store for semantic search
            ref = self.context_manager.get_file(filepath)
            if ref:
                self.vector_store.add_file(ref.filepath, ref.content)

        return success, message

    def remove_context_file(self, filepath: str) -> tuple[bool, str]:
        """
        Remove a file from context.

        Args:
            filepath: Path to the file

        Returns:
            Tuple of (success, message)
        """
        if not self.context_manager:
            return False, "Context management is not enabled"

        success, message = self.context_manager.remove_file(filepath)

        if success and self.vector_store:
            self.vector_store.remove_file(filepath)

        return success, message

    def list_context_files(self) -> list:
        """
        List all files currently in context.

        Returns:
            List of FileState objects
        """
        if not self.context_manager:
            return []

        return self.context_manager.list_files()

    def get_context_stats(self) -> Optional[Any]:
        """
        Get current context statistics.

        Returns:
            ContextStats object or None
        """
        if not self.context_manager:
            return None

        return self.context_manager.get_stats()

    def search_context(self, query: str, n_results: int = 5) -> list:
        """
        Semantically search through context files.

        Args:
            query: Search query
            n_results: Number of results to return

        Returns:
            List of SearchResult objects
        """
        if not self.vector_store:
            return []

        return self.vector_store.search(query, n_results=n_results)

    def get_formatted_context(self) -> str:
        """
        Get all context files formatted for LLM inclusion.

        Returns:
            Formatted string with all file contents
        """
        if not self.context_manager:
            return ""

        return self.context_manager.format_for_context()

    def clear_context(self) -> None:
        """Clear all files from context."""
        if self.context_manager:
            self.context_manager.clear()

        if self.vector_store:
            self.vector_store.clear()

    def stream(self, user_input: str, thread_id: str = "default", show_trace: bool = False):
        """
        Stream response for user input.

        Args:
            user_input: The user's query
            thread_id: Session thread identifier
            show_trace: Whether to show reasoning trace information
        """
        from langchain_core.messages import ToolMessage

        # Build messages with context injection
        messages = [
            SystemMessage(content=self.system_prompt),
            # Inject shared tool guide so all strategies see the same context
            SystemMessage(content=build_tool_guide(self.tools)),
        ]

        # Inject file context if available
        if self.context_manager:
            formatted_context = self.get_formatted_context()
            if formatted_context:
                messages.append(SystemMessage(content=f"FILE CONTEXT:\n{formatted_context}"))

        # Add user input
        messages.append(HumanMessage(content=user_input))

        input_state = {
            "messages": messages
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
                    # If the AI is calling tools, notify the user
                    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                        for tool_call in last_message.tool_calls:
                            tool_name = tool_call.get("name", "unknown")
                            yield f"[TOOL: {tool_name}]\n"

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
