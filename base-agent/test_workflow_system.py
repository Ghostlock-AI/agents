"""
Test the new workflow system.

This tests:
1. Agent vs Workflow mode distinction
2. Building custom workflows
3. Using workflow templates
4. Sequential LLM pipeline execution
"""

import sys
sys.path.insert(0, 'src')

from dotenv import load_dotenv
load_dotenv()

from agent import Agent
from workflow_system import WorkflowBuilder, WorkflowTemplates


def test_agent_mode():
    """Test that agent mode still works (ReAct)."""
    print("=" * 60)
    print("TEST 1: Agent Mode (ReAct)")
    print("=" * 60)

    agent = Agent(mode="agent", reasoning_strategy="react")
    print(f"Mode: {agent.mode}")
    print(f"Strategy: {agent.get_current_strategy_name()}")

    query = "What day is today?"
    print(f"\nQuery: {query}\n")

    for chunk in agent.stream(query, thread_id="test_agent"):
        print(chunk, end='', flush=True)

    print("\n✅ Agent mode works\n")


def test_workflow_template():
    """Test using a pre-built workflow template."""
    print("=" * 60)
    print("TEST 2: Workflow Mode (Template)")
    print("=" * 60)

    agent = Agent(mode="workflow")
    agent.load_workflow_template("research_and_summarize")

    print(f"Mode: {agent.mode}")
    print(f"Workflow: {agent.current_workflow.name}")
    print(f"Steps: {len(agent.current_workflow.steps)}")

    query = "What is LangGraph and why is it useful?"
    print(f"\nQuery: {query}\n")

    for chunk in agent.stream(query, thread_id="test_workflow", show_trace=True):
        print(chunk, end='', flush=True)

    print("\n✅ Workflow template works\n")


def test_custom_workflow():
    """Test building a custom workflow."""
    print("=" * 60)
    print("TEST 3: Custom Workflow")
    print("=" * 60)

    agent = Agent(mode="workflow")

    # Build a simple 2-step workflow
    workflow = (WorkflowBuilder("date_explainer")
                .add_step("get_date", "Use shell_exec tool to get today's date in format YYYY-MM-DD")
                .add_step("explain", "Explain what day of the week it is and any interesting facts about this date")
                .set_synthesizer("Format the date information in a friendly, conversational way")
                .build())

    agent.set_workflow(workflow)

    print(f"Mode: {agent.mode}")
    print(f"Workflow: {workflow.name}")
    print(f"Steps: {len(workflow.steps)}")

    query = "Tell me about today's date"
    print(f"\nQuery: {query}\n")

    for chunk in agent.stream(query, thread_id="test_custom"):
        print(chunk, end='', flush=True)

    # Show trace
    print("\n\n[Execution Trace]")
    for entry in agent.get_workflow_trace():
        print(f"  {entry['step']}: {entry['name']}")
        if 'output' in entry:
            print(f"    Output: {entry['output'][:100]}...")

    print("\n✅ Custom workflow works\n")


if __name__ == "__main__":
    print("\n🧪 Testing Workflow System\n")

    try:
        # Test 1: Agent mode (existing functionality)
        test_agent_mode()

        # Test 2: Workflow mode with template
        test_workflow_template()

        # Test 3: Custom workflow
        test_custom_workflow()

        print("=" * 60)
        print("🎉 ALL TESTS PASSED")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
