"""
Plan-and-Execute Strategy

Hybrid approach combining planning with adaptive execution:
1. Create high-level plan with multiple steps
2. Execute steps sequentially
3. Re-plan if needed based on results
4. Continue until goal achieved

Benefits:
- Structured approach for complex tasks
- Can adapt plan based on results (unlike ReWOO)
- Better tracking of progress
- Clearer reasoning trace

Drawbacks:
- Slower than ReWOO (sequential execution)
- More LLM calls than ReAct
- Can get stuck in planning loops

Best for:
- Multi-step tasks with clear goals
- Tasks requiring adaptive planning
- Complex projects needing decomposition
- Tasks where progress tracking matters

Research: Based on LangChain's PlanAndExecute pattern
"""

from typing import List, Literal, Union
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from .base import ReasoningStrategy


class Step(BaseModel):
    """A single step in a plan."""
    description: str = Field(description="What needs to be done in this step")
    completed: bool = Field(default=False, description="Whether this step is done")
    result: str = Field(default="", description="Result of executing this step")


class TaskPlan(BaseModel):
    """A plan with multiple steps."""
    goal: str = Field(description="The overall goal to achieve")
    steps: List[Step] = Field(description="List of steps to complete")
    current_step_index: int = Field(default=0, description="Index of current step")


class PlanExecuteStrategy(ReasoningStrategy):
    """Plan-and-Execute strategy implementation."""

    def __init__(self, max_replans: int = 3):
        """
        Initialize Plan-and-Execute strategy.

        Args:
            max_replans: Maximum number of times to replan (prevents infinite loops)
        """
        self.max_replans = max_replans
        self._replan_count = 0
        self._current_plan = None

    def get_name(self) -> str:
        return "plan-execute"

    def get_description(self) -> str:
        return (
            "Plan-and-Execute: Creates a high-level plan, executes steps sequentially, "
            "and replans if needed based on results. "
            "Best for: complex multi-step tasks, adaptive planning, progress tracking."
        )

    def get_config(self) -> dict:
        return {
            "max_replans": self.max_replans,
            "replan_count": self._replan_count,
            "current_plan": self._current_plan,
        }

    def update_config(self, **kwargs) -> None:
        if "max_replans" in kwargs:
            self.max_replans = kwargs["max_replans"]

    def create_graph(self, agent_state_class, llm_with_tools, tools):
        """Create the Plan-and-Execute reasoning graph."""

        # Prompt for creating initial plan
        planning_prompt = """You are a strategic planner. Break down this task into clear, actionable steps.

Task: {task}

Create a step-by-step plan. Each step should be:
1. Specific and actionable
2. Achievable with available tools
3. Build on previous steps

Available tools:
{tools}

Provide a numbered list of steps (3-7 steps recommended)."""

        # Prompt for execution
        execution_prompt = """You are executing step {step_num} of a plan.

Current Step: {step_description}

Previous Steps Completed:
{previous_results}

Execute this step using available tools as needed. Be thorough and precise."""

        # Prompt for checking if replanning is needed
        replan_check_prompt = """You are reviewing progress on a multi-step plan.

Original Goal: {goal}

Steps Completed:
{completed_steps}

Current Situation:
{current_result}

Remaining Steps:
{remaining_steps}

Do we need to replan? Answer with:
- CONTINUE: If we can proceed with the existing plan
- REPLAN: If we need to adjust the plan based on new information
- DONE: If the goal is already achieved

Then explain your reasoning."""

        def planner_node(state):
            """Create initial plan."""
            messages = state["messages"]

            # Get the user's task
            task = None
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage):
                    task = msg.content
                    break

            if not task:
                task = str(messages[-1].content)

            # Format tool descriptions
            tool_descriptions = "\n".join([
                f"- {tool.name}: {tool.description}"
                for tool in tools
            ])

            # Create planning prompt
            prompt_text = planning_prompt.format(
                task=task,
                tools=tool_descriptions
            )

            # Get plan from LLM
            planning_messages = [
                SystemMessage(content=prompt_text),
                HumanMessage(content=f"Create a plan for: {task}")
            ]

            response = llm_with_tools.invoke(planning_messages)

            # Store the plan (parse from response)
            # For simplicity, we'll just store the plan text
            self._current_plan = response.content

            # Add a message indicating we've created a plan
            plan_message = AIMessage(
                content=f"[PLAN CREATED]\n\n{response.content}"
            )

            return {"messages": [plan_message]}

        def executor_node(state):
            """Execute the current step."""
            messages = state["messages"]

            # Find the plan and determine current step
            # This is simplified - in production, you'd parse the plan more carefully
            execution_message = SystemMessage(
                content="Execute the next step of the plan using available tools."
            )

            # Invoke LLM with tools to execute
            response = llm_with_tools.invoke(messages + [execution_message])

            return {"messages": [response]}

        def replan_checker_node(state):
            """Check if we need to replan."""
            messages = state["messages"]

            # Create replan check prompt
            check_prompt = """Review the progress and determine if we should:
            1. CONTINUE with the current plan
            2. REPLAN with adjusted steps
            3. DONE (goal achieved)

            Respond with just the decision word first, then explanation."""

            check_message = SystemMessage(content=check_prompt)
            response = llm_with_tools.invoke(messages + [check_message])

            return {"messages": [response]}

        def should_continue(state) -> Literal["execute", "tools", "replan_check", "end"]:
            """Determine next action based on state."""
            messages = state["messages"]
            last_message = messages[-1]

            # If there are tool calls, go to tools
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"

            # Check if we just created a plan
            if isinstance(last_message, AIMessage) and "[PLAN CREATED]" in last_message.content:
                return "execute"

            # Check if we need to verify plan status
            # Look for execution results
            content = str(last_message.content).upper()

            if "DONE" in content or "GOAL ACHIEVED" in content:
                return "end"

            if "REPLAN" in content and self._replan_count < self.max_replans:
                self._replan_count += 1
                return "replan_check"

            # Check if we have recent tool use - if so, check plan status
            recent_tool_use = any(
                hasattr(msg, "type") and msg.type == "tool"
                for msg in messages[-3:]
            )

            if recent_tool_use:
                return "replan_check"

            # Default: continue executing
            return "execute"

        # Create the graph
        workflow = StateGraph(agent_state_class)

        # Add nodes
        workflow.add_node("planner", planner_node)
        workflow.add_node("executor", executor_node)
        workflow.add_node("tools", ToolNode(tools))
        workflow.add_node("replan_checker", replan_checker_node)

        # Define the flow
        workflow.add_edge(START, "planner")

        # After planning, start executing
        workflow.add_edge("planner", "executor")

        # After execution, decide next action
        workflow.add_conditional_edges(
            "executor",
            should_continue,
            {
                "execute": "executor",
                "tools": "tools",
                "replan_check": "replan_checker",
                "end": END,
            }
        )

        # After tools, go back to executor
        workflow.add_edge("tools", "executor")

        # After replan check, decide what to do
        workflow.add_conditional_edges(
            "replan_checker",
            should_continue,
            {
                "execute": "executor",
                "replan_check": "planner",  # Replan by going back to planner
                "end": END,
                "tools": "tools",
            }
        )

        # Add memory
        memory = MemorySaver()

        return workflow.compile(checkpointer=memory)

    def get_trace_info(self, state=None) -> dict:
        """Get trace information."""
        info = super().get_trace_info(state)
        info.update({
            "replan_count": self._replan_count,
            "max_replans": self.max_replans,
            "current_plan": self._current_plan,
        })
        return info
