"""
ReAct (Reason + Act) Reasoning Strategy

The classic iterative reasoning pattern:
1. Thought: Agent thinks about what to do
2. Action: Agent decides to use a tool OR respond directly
3. Observation: Results from tool are observed
4. Repeat until agent has final answer

Best for:
- General-purpose tasks with unknown complexity
- Tasks requiring iterative exploration
- Dynamic problem-solving where next steps depend on results

Performance: Fast, reliable, industry standard
"""

from typing import Literal
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from .base import ReasoningStrategy


class ReActStrategy(ReasoningStrategy):
    """ReAct (Reason + Act) reasoning strategy implementation."""

    def __init__(self, max_iterations: int = 20):
        """
        Initialize ReAct strategy.

        Args:
            max_iterations: Maximum number of thought-action cycles (prevents infinite loops)
        """
        self.max_iterations = max_iterations
        self._iteration_count = 0

    def get_name(self) -> str:
        return "react"

    def get_description(self) -> str:
        return (
            "ReAct (Reason + Act): Iterative reasoning with tool use. "
            "Agent thinks, acts, observes results, and repeats until done. "
            "Best for: general tasks, exploration, dynamic problem-solving."
        )

    def get_config(self) -> dict:
        return {
            "max_iterations": self.max_iterations,
            "current_iteration": self._iteration_count,
        }

    def update_config(self, **kwargs) -> None:
        if "max_iterations" in kwargs:
            self.max_iterations = kwargs["max_iterations"]

    def create_graph(self, agent_state_class, llm_with_tools, tools):
        """Create the ReAct reasoning graph."""

        def should_continue(state) -> Literal["tools", "end"]:
            """
            Determine if we should continue to tools or end.

            After the agent runs, check if it wants to use tools.
            If tool_calls present -> route to tools
            If no tool_calls -> route to end (agent has final answer)
            """
            messages = state["messages"]
            last_message = messages[-1]

            # Check iteration limit
            if self._iteration_count >= self.max_iterations:
                return "end"

            # If LLM makes tool calls, continue to tools
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"

            # Otherwise end - agent has final response
            return "end"

        def agent_node(state):
            """The agent thinks and decides on actions."""
            messages = state["messages"]
            self._iteration_count += 1

            # Call LLM (which has tools bound)
            response = llm_with_tools.invoke(messages)

            return {"messages": [response]}

        # Create the graph
        workflow = StateGraph(agent_state_class)

        # Add nodes
        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", ToolNode(tools))

        # Define the flow
        workflow.add_edge(START, "agent")

        # Conditional edge: after agent, decide what to do
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            {
                "tools": "tools",
                "end": END,
            }
        )

        # After tools execute, always go back to agent
        workflow.add_edge("tools", "agent")

        # Add memory for conversation history
        memory = MemorySaver()

        # Compile and return
        return workflow.compile(checkpointer=memory)

    def get_trace_info(self, state=None) -> dict:
        """Get trace information for debugging."""
        info = super().get_trace_info(state)
        info.update({
            "iteration": self._iteration_count,
            "max_iterations": self.max_iterations,
        })

        if state and "messages" in state:
            last_msg = state["messages"][-1] if state["messages"] else None
            if last_msg:
                info["last_action"] = "tool_call" if hasattr(last_msg, "tool_calls") and last_msg.tool_calls else "response"

        return info
