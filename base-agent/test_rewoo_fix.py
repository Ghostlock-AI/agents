"""Quick test of ReWOO fixes."""
import subprocess
import sys

def test_rewoo_query(query: str) -> str:
    """Test a ReWOO query and return output."""
    # First switch to rewoo, then run query
    result = subprocess.run(
        ["python3", "src/main.py", query],
        capture_output=True,
        text=True,
        timeout=45,
        env={**subprocess.os.environ, "REWOO_STRATEGY": "true"}
    )
    return result.stdout.strip()

print("Testing ReWOO fixes...")
print("=" * 60)

print("\nTest 1: Date query")
print("-" * 60)
output = test_rewoo_query("What day is today?")
print(output)
has_tool = "[TOOL:" in output
has_plan = "[PLAN CREATED]" in output
has_answer = len(output) > 50
print(f"\n✓ Has plan: {has_plan}")
print(f"✓ Has tool: {has_tool}")
print(f"✓ Has answer: {has_answer}")

if has_plan and has_tool and has_answer:
    print("\n✅ Test 1 PASSED")
else:
    print("\n❌ Test 1 FAILED")
    sys.exit(1)

print("\n" + "=" * 60)
print("Test 2: Research query")
print("-" * 60)
output = test_rewoo_query("When was the last time Trump visited Australia?")
print(output)
has_tool = "[TOOL:" in output
has_plan = "[PLAN CREATED]" in output
has_answer = len(output) > 100
print(f"\n✓ Has plan: {has_plan}")
print(f"✓ Has tool: {has_tool}")
print(f"✓ Has answer: {has_answer}")

if has_plan and has_tool and has_answer:
    print("\n✅ Test 2 PASSED")
else:
    print("\n❌ Test 2 FAILED")
    sys.exit(1)

print("\n" + "=" * 60)
print("🎉 ALL REWOO TESTS PASSED")
print("=" * 60)
