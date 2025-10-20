"""
Agent - Core agent class with reasoning loop and memory.
"""

import os
from typing import TypedDict, Annotated
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver


class AgentState(TypedDict):
    """State that tracks conversation messages."""
    messages: Annotated[list[BaseMessage], add_messages]


class Agent:
    """Base agent with LLM, reasoning loop, and memory."""

    def __init__(
        self,
        model_name: str = None,
        temperature: float = 0.7,
        system_prompt_path: str = "system_prompt.txt"
    ):
        """Initialize the agent."""
        self.model_name = model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = temperature
        self.system_prompt = self._load_system_prompt(system_prompt_path)

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            streaming=True
        )

        # Build the reasoning graph
        self.graph = self._build_graph()

    def _load_system_prompt(self, path: str) -> str:
        """Load system prompt from file."""
        # Path relative to this file (in src/)
        prompt_path = Path(__file__).parent / path

        if prompt_path.exists():
            return prompt_path.read_text().strip()

        # Default fallback
        return "You are a helpful AI assistant."

    def _agent_node(self, state: AgentState) -> AgentState:
        """Process messages and generate response."""
        messages = state["messages"]

        # Prepend system message
        messages_with_system = [
            {"role": "system", "content": self.system_prompt}
        ] + [msg for msg in messages]

        # Call LLM
        response = self.llm.invoke(messages_with_system)

        return {"messages": [response]}

    def _build_graph(self):
        """Build the LangGraph reasoning graph."""
        workflow = StateGraph(AgentState)

        # Add agent node
        workflow.add_node("agent", self._agent_node)

        # Set up flow
        workflow.add_edge(START, "agent")
        workflow.add_edge("agent", END)

        # Add memory
        memory = MemorySaver()

        # Compile with checkpointing
        return workflow.compile(checkpointer=memory)

    def stream(self, user_input: str, thread_id: str = "default"):
        """Stream response for user input."""
        from langchain_core.messages import HumanMessage, AIMessage

        input_state = {
            "messages": [HumanMessage(content=user_input)]
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

                if isinstance(last_message, AIMessage):
                    yield last_message.content
