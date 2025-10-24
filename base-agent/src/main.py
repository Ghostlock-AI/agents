#!/usr/bin/env python3
"""
Base Agent - Main Entry Point

Usage:
    python main.py                      # Interactive TUI mode
    python main.py "your question here" # One-shot query
    python main.py "what is X?" "tell me about Y"  # Multi-part query
"""

import os
import sys

from dotenv import load_dotenv

import cli
from agent import Agent


def main():
    """Initialize and run the agent."""
    # Load environment
    load_dotenv()

    # Validate API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables")
        print("Please create a .env file with your OpenAI API key")
        sys.exit(1)

    # Create agent with defaults
    agent = Agent()

    # Run CLI with arguments
    # If args provided: one-shot mode
    # If no args: interactive TUI mode with file attachment support
    cli.run(agent, sys.argv[1:])


if __name__ == "__main__":
    main()
