"""
Test ReAct Strategy Improvements

This test suite validates that the agent:
1. Always uses tools for time-sensitive information
2. Never hallucinates current dates/times from training data
3. Provides well-formatted, comprehensive but brief answers
4. Correctly decides when tools are needed vs using existing knowledge
"""

import subprocess
import sys


def run_query(query: str) -> tuple[str, bool]:
    """Run a query and return (output, used_tools)."""
    result = subprocess.run(
        ["python3", "src/main.py", query],
        capture_output=True,
        text=True,
        timeout=30
    )
    output = result.stdout.strip()
    used_tools = "[TOOL:" in output
    return output, used_tools


def test_date_queries():
    """Test that date queries always use tools."""
    print("Testing date queries...")
    queries = [
        "What's today's date?",
        "I need to know today's date",
        "Current date please",
        "Tell me today's date",
        "What date is today?",
    ]

    for query in queries:
        output, used_tools = run_query(query)
        assert used_tools, f"FAIL: '{query}' did not use tools"
        assert "2023" not in output, f"FAIL: '{query}' returned training data date"
        print(f"  ✓ {query}")

    print("✅ All date queries use tools correctly\n")


def test_time_queries():
    """Test that time queries always use tools."""
    print("Testing time queries...")
    queries = [
        "What time is it?",
        "What day is it?",
        "What day of the week is it?",
    ]

    for query in queries:
        output, used_tools = run_query(query)
        assert used_tools, f"FAIL: '{query}' did not use tools"
        print(f"  ✓ {query}")

    print("✅ All time queries use tools correctly\n")


def test_current_events():
    """Test that current event queries use tools."""
    print("Testing current event queries...")
    queries = [
        "What's the weather in San Francisco?",
        "Who won the latest NBA championship?",
        "Tell me about recent tech news",
    ]

    for query in queries:
        output, used_tools = run_query(query)
        assert used_tools, f"FAIL: '{query}' did not use tools"
        print(f"  ✓ {query}")

    print("✅ All current event queries use tools correctly\n")


def test_knowledge_queries():
    """Test that knowledge queries work without tools when appropriate."""
    print("Testing knowledge-based queries...")
    queries = [
        "Explain how React hooks work",
        "What is the difference between Python lists and tuples?",
    ]

    for query in queries:
        output, used_tools = run_query(query)
        # These should provide answers (tools optional)
        assert len(output) > 100, f"FAIL: '{query}' gave too short answer"
        print(f"  ✓ {query} (tools: {used_tools})")

    print("✅ Knowledge queries answered appropriately\n")


def test_answer_format():
    """Test that answers are well-formatted."""
    print("Testing answer formatting...")

    # Test brief answer for simple query
    output, _ = run_query("What's today's date?")
    lines = output.split("\n")
    # Should be brief (a few lines max)
    assert len(lines) <= 5, "FAIL: Date query too verbose"
    print("  ✓ Simple queries are brief")

    # Test comprehensive answer for complex query
    output, _ = run_query("What's the weather in San Francisco?")
    # Should have structure (bullet points or sections)
    assert "•" in output or "-" in output or "\n" in output, "FAIL: Weather not formatted"
    print("  ✓ Complex queries are well-formatted")

    print("✅ Answer formatting is appropriate\n")


if __name__ == "__main__":
    print("=" * 60)
    print("ReAct Strategy Improvement Tests")
    print("=" * 60 + "\n")

    try:
        test_date_queries()
        test_time_queries()
        test_current_events()
        test_knowledge_queries()
        test_answer_format()

        print("=" * 60)
        print("🎉 ALL TESTS PASSED")
        print("=" * 60)
        sys.exit(0)

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        sys.exit(1)
