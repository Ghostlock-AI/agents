#!/usr/bin/env python3
"""
Test script for reasoning strategies.

This script demonstrates:
1. Listing available strategies
2. Switching between strategies
3. Getting strategy info
"""

import os
from dotenv import load_dotenv

# Add src to path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from agent import Agent

def main():
    # Load environment
    load_dotenv()

    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ Error: OPENAI_API_KEY not found in environment")
        print("Please create a .env file with your OpenAI API key")
        return

    print("=" * 70)
    print("REASONING STRATEGIES TEST")
    print("=" * 70)
    print()

    # Create agent
    print("Initializing agent...")
    agent = Agent()
    print(f"✓ Agent initialized with strategy: {agent.get_current_strategy_name()}")
    print()

    # List all strategies
    print("-" * 70)
    print("AVAILABLE STRATEGIES:")
    print("-" * 70)
    strategies = agent.list_strategies()
    for strategy in strategies:
        marker = "✓" if strategy["is_current"] else " "
        print(f"[{marker}] {strategy['name']}")
        print(f"    {strategy['description']}")
        print()

    # Test switching strategies
    print("-" * 70)
    print("TESTING STRATEGY SWITCHING:")
    print("-" * 70)

    test_strategies = ["react", "rewoo", "plan-execute", "lats"]

    for strategy_name in test_strategies:
        print(f"\nSwitching to: {strategy_name}")
        try:
            agent.switch_reasoning_strategy(strategy_name)
            info = agent.get_strategy_info()
            print(f"  ✓ Switched successfully")
            print(f"  Strategy: {info['name']}")
            print(f"  Supports streaming: {info['supports_streaming']}")
            if info['config']:
                print(f"  Config: {info['config']}")
        except Exception as e:
            print(f"  ❌ Error: {e}")

    print()
    print("-" * 70)
    print("TEST COMPLETE")
    print("-" * 70)
    print()
    print("To use the agent interactively, run:")
    print("  python src/main.py")
    print()
    print("Available commands in the TUI:")
    print("  /reasoning list           - List all strategies")
    print("  /reasoning current        - Show current strategy")
    print("  /reasoning switch <name>  - Switch to a different strategy")
    print("  /reasoning info [name]    - Show detailed strategy info")
    print()


if __name__ == "__main__":
    main()
