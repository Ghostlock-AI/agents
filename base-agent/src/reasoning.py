"""
Reasoning - ReAct (Reason + Act) reasoning loop implementation.

ReAct Pattern:
1. Thought: Agent thinks about what to do
2. Action: Agent decides to use a tool OR respond directly
3. Observation: Results from tool are observed
4. Repeat until agent has final answer

LangGraph automatically handles this loop when we:
- Bind tools to the LLM
- Add a ToolNode to execute tool calls
- Set up conditional edges (continue if tools called, end if final answer)
"""

from typing import Literal
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver


def should_continue(state) -> Literal["tools", "end"]:
    """
    Determine if we should continue to tools or end.

    After the agent runs, check if it wants to use tools.
    If tool_calls present -> route to tools
    If no tool_calls -> route to end (agent has final answer)
    """
    messages = state["messages"]
    last_message = messages[-1]

    # If LLM makes tool calls, continue to tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # Otherwise end - agent has final response
    return "end"


def create_react_graph(agent_state_class, llm_with_tools, tools):
    """
    Create a ReAct reasoning graph.

    Args:
        agent_state_class: TypedDict defining the state schema
        llm_with_tools: LLM with tools bound via llm.bind_tools()
        tools: List of tool instances

    Returns:
        Compiled graph ready to execute

    Graph Structure:
        START -> agent -> [should_continue]
                            |
                            +-> tools -> agent (loop back)
                            |
                            +-> END
    """

    # Define the agent node
    def agent_node(state):
        """The agent thinks and decides on actions."""
        messages = state["messages"]

        # Call LLM (which has tools bound)
        # LLM will either:
        # 1. Return a normal response (done)
        # 2. Return a tool_call request (needs to use a tool)
        response = llm_with_tools.invoke(messages)

        return {"messages": [response]}

    # Create the graph
    workflow = StateGraph(agent_state_class)

    # Add nodes
    workflow.add_node("agent", agent_node)

    # ToolNode automatically executes any tool calls from the agent
    # It looks at the last message, finds tool_calls, executes them,
    # and returns ToolMessages with results
    workflow.add_node("tools", ToolNode(tools))

    # Define the flow
    workflow.add_edge(START, "agent")

    # Conditional edge: after agent, decide what to do
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",  # If tools needed, go to tools
            "end": END         # If done, end
        }
    )

    # After tools execute, always go back to agent
    # Agent will see the tool results and continue reasoning
    workflow.add_edge("tools", "agent")

    # Add memory for conversation history
    memory = MemorySaver()

    # Compile and return
    return workflow.compile(checkpointer=memory)


# ReAct Loop Explanation:
#
# Turn 1:
#   User: "What's the weather in Paris?"
#   -> agent: Thinks "I need to search for this"
#   -> agent outputs: tool_call(tavily_search, "weather Paris")
#   -> should_continue: sees tool_call, routes to "tools"
#   -> tools: Executes search, returns results
#   -> Back to agent with results
#
# Turn 2:
#   -> agent: Sees tool results in message history
#   -> agent outputs: "Based on the search, the weather in Paris is..."
#   -> should_continue: no tool_calls, routes to "end"
#   -> END
#
# The agent can use tools multiple times in one query if needed!
# Each loop through adds context, building up to the final answer.
