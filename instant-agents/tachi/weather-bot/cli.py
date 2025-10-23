#!/usr/bin/env python3
"""
Interactive CLI for the smolagent.
Provides a classic chat interface with input/output loop.
"""

import sys
from agent import create_agent


def print_banner():
    """Print welcome banner."""
    print("=" * 60)
    print("HuggingFace Smolagent CLI")
    print("=" * 60)
    print("Type your requests and press Enter.")
    print("Type 'exit', 'quit', or press Ctrl+C to exit.")
    print("=" * 60)
    print()


def main():
    """Run the interactive CLI loop."""
    try:
        # Initialize agent once at startup
        print("Initializing agent...")
        agent = create_agent()
        print("Agent ready!\n")

        print_banner()

        # Main interaction loop
        while True:
            try:
                # Get user input
                user_input = input("\nYou: ").strip()

                # Check for exit commands
                if user_input.lower() in ["exit", "quit", "q"]:
                    print("\nGoodbye!")
                    break

                # Skip empty inputs
                if not user_input:
                    continue

                # Run agent with user input
                print("\nAgent: ", end="", flush=True)
                result = agent.run(user_input)
                print(result)

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except EOFError:
                print("\n\nGoodbye!")
                break

    except Exception as e:
        print(f"\nError initializing agent: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
