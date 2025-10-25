"""
ReWOO (Reasoning WithOut Observation) Strategy

Advanced planning strategy that:
1. Plans ALL steps upfront (without executing)
2. Executes all tool calls in parallel
3. Synthesizes final answer from all results

Benefits:
- Massive speedup via parallel execution
- Better for predictable/structured tasks
- Reduces token usage (fewer LLM calls)

Drawbacks:
- Can't adapt plan based on tool results
- Requires clear problem structure upfront
- Less effective for exploratory tasks

Best for:
- Research tasks (multiple searches in parallel)
- Data gathering from multiple sources
- Tasks with clear, predictable steps

Research: https://arxiv.org/abs/2305.18323
"""

from typing import List, Dict, Any, Literal
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from .base import ReasoningStrategy


class Plan(BaseModel):
    """A plan with multiple steps to execute."""
    steps: List[Dict[str, Any]] = Field(
        description="List of steps, each with 'tool', 'args', and 'depends_on' keys"
    )


class ReWOOStrategy(ReasoningStrategy):
    """ReWOO (Reasoning Without Observation) strategy implementation."""

    def __init__(self):
        """Initialize ReWOO strategy."""
        self._plan = None
        self._execution_results = {}

    def get_name(self) -> str:
        return "rewoo"

    def get_description(self) -> str:
        return (
            "ReWOO (Reasoning Without Observation): Plans all steps upfront, "
            "then executes tools in parallel for maximum speed. "
            "Best for: research tasks, data gathering, predictable workflows."
        )

    def create_graph(self, agent_state_class, llm_with_tools, tools):
        """Create the ReWOO reasoning graph."""

        # Create a planning prompt
        planning_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a strategic planner. Given a user query, create a detailed plan of steps to solve it.

For each step, specify:
1. The tool to use
2. The arguments for that tool
3. Which previous steps this depends on (empty list if independent)

Available tools: {tools}

Return your plan as a structured list of steps. Think carefully about which steps can run in parallel (no dependencies) vs which must run sequentially.

Example plan format:
Step 1: {{tool: "search", args: {{query: "Python asyncio"}}, depends_on: []}}
Step 2: {{tool: "search", args: {{query: "Python threading"}}, depends_on: []}}  # Can run parallel with Step 1
Step 3: {{tool: "web_fetch", args: {{url: "result from step 1"}}, depends_on: [1]}}  # Must wait for Step 1
"""),
            ("human", "{query}")
        ])

        def planner_node(state):
            """Create a plan for solving the task."""
            messages = state["messages"]
            last_message = messages[-1]

            # Get the user query
            if isinstance(last_message, HumanMessage):
                query = last_message.content
            else:
                # Find the last human message
                for msg in reversed(messages):
                    if isinstance(msg, HumanMessage):
                        query = msg.content
                        break
                else:
                    query = str(last_message.content)

            # Create plan prompt
            tool_descriptions = "\n".join([
                f"- {tool.name}: {tool.description}"
                for tool in tools
            ])

            prompt = planning_prompt.format_messages(
                tools=tool_descriptions,
                query=query
            )

            # Get plan from LLM
            response = llm_with_tools.invoke(prompt)

            # Store the plan in state
            # The plan will be in the AI response
            return {"messages": [response]}

        def executor_node(state):
            """Execute the planned steps (in parallel where possible)."""
            messages = state["messages"]
            last_message = messages[-1]

            # Check if this is a plan or if we need to execute tools
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                # This is handled by ToolNode automatically
                return {"messages": []}

            # If no tool calls, we're done planning - ask agent to execute
            # by moving to tools node
            return {"messages": [last_message]}

        def synthesizer_node(state):
            """Synthesize final answer from all execution results."""
            messages = state["messages"]

            # Find all tool results
            tool_results = []
            for msg in messages:
                if hasattr(msg, "type") and msg.type == "tool":
                    tool_results.append(msg.content)

            # Create synthesis prompt
            synthesis_prompt = f"""Based on the following tool execution results, provide a comprehensive answer to the original query:

Tool Results:
{chr(10).join([f"- {result}" for result in tool_results])}

Provide a clear, well-structured answer that synthesizes these results."""

            # Get final response
            response = llm_with_tools.invoke(
                messages + [SystemMessage(content=synthesis_prompt)]
            )

            return {"messages": [response]}

        def should_continue(state) -> Literal["tools", "synthesize", "end"]:
            """Route based on current state."""
            messages = state["messages"]
            last_message = messages[-1]

            # If we have tool calls, execute them
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"

            # Check if we have tool results to synthesize
            has_tool_results = any(
                hasattr(msg, "type") and msg.type == "tool"
                for msg in messages
            )

            if has_tool_results:
                # Check if we've already synthesized
                # Look for a final AI message after tool results
                for i in range(len(messages) - 1, -1, -1):
                    if hasattr(messages[i], "type") and messages[i].type == "tool":
                        # Found last tool message, check if there's an AI message after
                        if i < len(messages) - 1 and isinstance(messages[-1], AIMessage):
                            return "end"
                        return "synthesize"

            return "end"

        # Create the graph
        workflow = StateGraph(agent_state_class)

        # Add nodes
        workflow.add_node("planner", planner_node)
        workflow.add_node("tools", ToolNode(tools))
        workflow.add_node("synthesizer", synthesizer_node)

        # Define the flow
        workflow.add_edge(START, "planner")

        # After planning, decide what to do
        workflow.add_conditional_edges(
            "planner",
            should_continue,
            {
                "tools": "tools",
                "synthesize": "synthesizer",
                "end": END,
            }
        )

        # After tools, synthesize
        workflow.add_edge("tools", "synthesizer")

        # After synthesis, end
        workflow.add_edge("synthesizer", END)

        # Add memory
        memory = MemorySaver()

        return workflow.compile(checkpointer=memory)

    def get_trace_info(self, state=None) -> dict:
        """Get trace information."""
        info = super().get_trace_info(state)

        if self._plan:
            info["plan"] = self._plan

        if self._execution_results:
            info["execution_results"] = self._execution_results

        return info
