"""
CLI - Command-line interface for agent interaction.
"""


def print_header():
    """Print CLI header."""
    print("=" * 60)
    print("  Base Agent - Chat Interface")
    print("=" * 60)
    print()
    print("Type 'exit', 'quit', or 'q' to end the conversation")
    print()


def get_user_input() -> str:
    """Get input from user."""
    return input("> ").strip()


def print_response(agent, user_input: str, session_id: str = "main_session"):
    """Print streaming response from agent."""
    print("Assistant: ", end="", flush=True)

    for content in agent.stream(user_input, session_id):
        print(content, end="", flush=True)

    print()


def run(agent):
    """Run the CLI interaction loop."""
    print_header()

    session_id = "main_session"

    while True:
        try:
            user_input = get_user_input()

            # Check for exit
            if user_input.lower() in ["exit", "quit", "q"]:
                print("\nGoodbye!")
                break

            # Skip empty
            if not user_input:
                continue

            # Print response
            print_response(agent, user_input, session_id)
            print()

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.\n")
