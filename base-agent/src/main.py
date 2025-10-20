#!/usr/bin/env python3
"""
Base Agent - Main Entry Point
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

    # Run CLI, cli takes agent ref
    # to output the stream response
    cli.run(agent)


if __name__ == "__main__":
    main()
