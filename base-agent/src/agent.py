"""
Agent - Core agent class with reasoning loop and memory.
"""

import os
from typing import TypedDict, Annotated
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages

from tools import get_tools
from reasoning import create_react_graph


class AgentState(TypedDict):
    """State that tracks conversation messages."""
    messages: Annotated[list[BaseMessage], add_messages]


class Agent:
    """Base agent with LLM, ReAct reasoning loop, tools, and memory."""

    def __init__(
        self,
        model_name: str = None,
        temperature: float = 0.7,
        system_prompt_path: str = "system_prompt.txt",
        tools: list = None
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

        # Build the ReAct reasoning graph
        self.graph = self._build_graph()

    def _load_system_prompt(self, path: str) -> str:
        """Load system prompt from file."""
        prompt_path = Path(__file__).parent / path

        if prompt_path.exists():
            return prompt_path.read_text().strip()

        return "You are a helpful AI assistant."

    def _build_graph(self):
        """Build the ReAct reasoning graph with tools."""
        # Use the ReAct graph builder from reasoning.py
        return create_react_graph(AgentState, self.llm_with_tools, self.tools)

    def stream(self, user_input: str, thread_id: str = "default"):
        """Stream response for user input."""
        from langchain_core.messages import ToolMessage

        # Prepend system message to the user input
        input_state = {
            "messages": [
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=user_input)
            ]
        }

        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }

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

                    # Only yield if there's actual content (not just tool calls)
                    if last_message.content:
                        yield last_message.content

                # Show tool results
                elif isinstance(last_message, ToolMessage):
                    # Don't display tool output, just let it feed back to agent
                    pass
