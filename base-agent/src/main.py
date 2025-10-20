#!/usr/bin/env python3
"""
Base Agent - Main Entry Point

A simple chat agent built with LangGraph that maintains conversation history.
"""

import os
import sys
from typing import TypedDict, Annotated
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver


# Load environment variables
load_dotenv()

# Validate API key
if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY not found in environment variables")
    print("Please create a .env file with your OpenAI API key")
    sys.exit(1)


# Define the state schema for our agent
class AgentState(TypedDict):
    """State that tracks conversation messages."""
    messages: Annotated[list[BaseMessage], add_messages]


# System prompt
SYSTEM_PROMPT = """You are a helpful AI assistant. You are:
- Concise and clear in your responses
- Able to maintain context across the conversation
- Helpful and informative

Respond naturally to user queries."""


def create_agent():
    """Create the LangGraph agent with session memory."""

    # Initialize the LLM
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    llm = ChatOpenAI(
        model=model_name,
        temperature=0.7,
        streaming=True
    )

    # Define the agent node
    def agent_node(state: AgentState) -> AgentState:
        """Process messages and generate response."""
        # Add system prompt to the beginning if this is the first message
        messages = state["messages"]

        # Prepend system message for context
        messages_with_system = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ] + [msg for msg in messages]

        # Call the LLM
        response = llm.invoke(messages_with_system)

        return {"messages": [response]}

    # Build the graph
    workflow = StateGraph(AgentState)

    # Add the agent node
    workflow.add_node("agent", agent_node)

    # Set up the flow
    workflow.add_edge(START, "agent")
    workflow.add_edge("agent", END)

    # Add memory for session persistence
    memory = MemorySaver()

    # Compile the graph with checkpointing
    app = workflow.compile(checkpointer=memory)

    return app


def stream_response(app, user_input: str, thread_id: str = "default"):
    """Stream the agent's response token by token."""

    # Create the input state
    input_state = {
        "messages": [HumanMessage(content=user_input)]
    }

    # Configuration for session tracking
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    # Stream the response
    print("Assistant: ", end="", flush=True)

    for chunk in app.stream(input_state, config, stream_mode="values"):
        # Get the last message from the chunk
        if "messages" in chunk and chunk["messages"]:
            last_message = chunk["messages"][-1]

            # Only stream AI messages
            if isinstance(last_message, AIMessage):
                # Print the content as it streams
                print(last_message.content, end="", flush=True)

    print()  # New line after response


def main():
    """Main chat loop."""

    print("=" * 60)
    print("  Base Agent - Chat Interface")
    print("=" * 60)
    print()
    print("Type 'exit', 'quit', or 'q' to end the conversation")
    print()

    # Create the agent
    agent = create_agent()

    # Session ID for memory persistence
    session_id = "main_session"

    # Chat loop
    while True:
        try:
            # Get user input
            user_input = input("> ").strip()

            # Check for exit commands
            if user_input.lower() in ["exit", "quit", "q"]:
                print("\nGoodbye!")
                break

            # Skip empty inputs
            if not user_input:
                continue

            # Stream the response
            stream_response(agent, user_input, session_id)
            print()

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.\n")


if __name__ == "__main__":
    main()
