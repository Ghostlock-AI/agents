#!/usr/bin/env python3
"""
Test Pattern Selector

Validates that the pattern selector correctly identifies task characteristics
and recommends appropriate reasoning patterns.
"""

import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from reasoning.pattern_selector import PatternSelector
from langchain_openai import ChatOpenAI


def print_separator():
    print("\n" + "=" * 80 + "\n")


def test_query(selector: PatternSelector, query: str, expected_pattern: str = None):
    """Test a single query and display results."""
    print(f"QUERY: {query}")
    print()

    recommendation = selector.select_pattern(query)

    print(f"Recommended Pattern: {recommendation.pattern_name} ({recommendation.pattern_type})")
    print(f"Confidence: {recommendation.confidence:.0%}")
    print(f"Reasoning: {recommendation.reasoning}")
    print()
    print("Characteristics:")
    for char, value in recommendation.characteristics.items():
        if value:
            print(f"  ✓ {char}")

    if expected_pattern:
        match = recommendation.pattern_name == expected_pattern
        status = "✓ CORRECT" if match else f"✗ EXPECTED {expected_pattern}"
        print()
        print(status)

    print_separator()

    return recommendation


def main():
    load_dotenv()

    # Initialize LLM and selector
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    selector = PatternSelector(llm)

    print("=" * 80)
    print("PATTERN SELECTOR TEST SUITE")
    print("=" * 80)

    # Test 1: Open-ended exploratory query → ReAct
    print("\nTest 1: Open-ended exploratory task")
    print_separator()
    test_query(
        selector,
        "Help me debug why my Python script is crashing. I'm not sure what's wrong.",
        expected_pattern="react"
    )

    # Test 2: Quality-critical content generation → Prompt Chain
    print("\nTest 2: Content generation with quality requirements")
    print_separator()
    test_query(
        selector,
        "Write a professional blog post about AI trends, make sure it's well-structured and accurate.",
        expected_pattern="prompt-chain"
    )

    # Test 3: Research task with parallel execution → ReWOO
    print("\nTest 3: Research task with multiple lookups")
    print_separator()
    test_query(
        selector,
        "Find the latest news about OpenAI and Anthropic, then summarize the key developments.",
        expected_pattern="rewoo"
    )

    # Test 4: Multi-step project planning → Plan-Execute
    print("\nTest 4: Complex multi-step project")
    print_separator()
    test_query(
        selector,
        "Create a complete Python web API project with authentication, database, and tests.",
        expected_pattern="plan-execute"
    )

    # Test 5: Simple current date query → ReAct
    print("\nTest 5: Simple information query")
    print_separator()
    test_query(
        selector,
        "What day is today?",
        expected_pattern="react"
    )

    # Test 6: Very complex problem → LATS
    print("\nTest 6: Complex optimization problem")
    print_separator()
    test_query(
        selector,
        "Optimize this sorting algorithm for the best possible time complexity. "
        "Consider all edge cases and provide multiple approaches.",
        expected_pattern="lats"
    )

    # Test 7: Multi-domain query → Routing
    print("\nTest 7: Query that could benefit from routing")
    print_separator()
    test_query(
        selector,
        "I need help with my code and also want to write some documentation.",
        expected_pattern="routing"
    )

    # Test 8: Simple coding question → ReAct
    print("\nTest 8: Straightforward coding question")
    print_separator()
    test_query(
        selector,
        "How do I read a JSON file in Python?",
        expected_pattern="react"
    )

    print("\n" + "=" * 80)
    print("TEST SUITE COMPLETE")
    print("=" * 80)
    print("\nNote: Expected patterns are based on Anthropic's guidance.")
    print("The LLM may make reasonable alternative choices depending on interpretation.")


if __name__ == "__main__":
    main()
