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

from typing import List, Literal, Union, Optional
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from reasoning.base import ReasoningStrategy
from reasoning.tool_context import build_tool_guide


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
        """Create the Plan-and-Execute reasoning graph with enforced step execution."""

        # Prompt for creating initial plan (strict JSON with tool + args per step)
        planning_prompt = """You are a strategic planner. Return STRICT JSON for a step-by-step plan.

TOOL CONTEXT (catalog, rules, examples):
{tool_guide}

JSON schema (example):
{{
  "steps": [
    {{"id": 1, "description": "Search for X", "tool": "ddgs_search", "args": {{"query": "X", "max_results": 5}}}},
    {{"id": 2, "description": "Fetch details", "tool": "web_fetch", "args": {{"url": "..."}}}}
  ]
}}

Rules:
- Use integers for id starting at 1.
- Each step must specify a tool and args matching the tool signature.
- Do NOT include any text outside JSON.
"""

        # No free-form execution prompt; we will emit explicit tool calls for each step.

        # Replanning omitted in this simplified, deterministic executor.

        def _tools_signature() -> str:
            sigs = []
            for t in tools:
                name = getattr(t, "name", "")
                doc = getattr(t, "description", "")
                sigs.append(f"- {name}: {doc}")
            return "\n".join(sigs)

        def _parse_json(text: str) -> Optional[dict]:
            import json, re
            s = text.strip()
            m = re.search(r"\{[\s\S]*\}$", s)
            if m:
                s = m.group(0)
            try:
                return json.loads(s)
            except Exception:
                return None

        def planner_node(state):
            """Create initial plan (structured) and initialize step index."""
            messages = state["messages"]
            task = None
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage):
                    task = msg.content
                    break
            if not task:
                task = str(messages[-1].content)

            prompt_text = planning_prompt.format(tool_guide=build_tool_guide(tools))
            planning_messages = [
                SystemMessage(content=prompt_text),
                HumanMessage(content=f"Task: {task}")
            ]
            response = llm_with_tools.invoke(planning_messages)
            text = response.content if isinstance(response.content, str) else str(response.content)
            plan_json = _parse_json(text) or {"steps": [
                {"id": 1, "description": f"Search: {task}", "tool": "ddgs_search", "args": {"query": task, "max_results": 5}}
            ]}
            self._current_plan = plan_json
            return {
                "messages": [AIMessage(content=f"[PLAN CREATED]\n\n{text}")],
                "plan": plan_json,
                "step_idx": 0,
                "results": [],
            }

        def step_to_calls_node(state):
            """Emit explicit tool call for the current step (sequential)."""
            plan = state.get("plan") or {"steps": []}
            steps: List[dict] = plan.get("steps", [])
            idx = int(state.get("step_idx") or 0)
            if idx >= len(steps):
                return {"messages": []}
            step = steps[idx]
            tool = step.get("tool")
            args = dict(step.get("args") or {})
            sid = step.get("id", idx + 1)
            # Heuristic: if web_fetch URL is not absolute, try to take top URL from the last ddgs_search tool output
            if tool == "web_fetch":
                url = str(args.get("url", ""))
                if not url.startswith(("http://", "https://")):
                    last_search: Optional[ToolMessage] = None
                    for m in reversed(state["messages"]):
                        if isinstance(m, ToolMessage) and getattr(m, "name", None) == "ddgs_search":
                            last_search = m
                            break
                    if last_search is not None:
                        import re
                        urls = re.findall(r"URL:\s*(\S+)", str(last_search.content))
                        if urls:
                            args["url"] = urls[0]
            if not tool:
                return {"messages": []}
            return {"messages": [AIMessage(content="", tool_calls=[{"name": tool, "args": args, "id": f"pe{sid}"}]) ]}

        # Replan checker removed in this simplified deterministic loop

        def after_planner_route(state) -> Literal["execute", "end"]:
            plan = state.get("plan") or {}
            steps = plan.get("steps", [])
            return "execute" if steps else "end"

        def after_execute_route(state) -> Literal["tools", "end"]:
            # If we emitted a tool call, go to tools; else end
            messages = state["messages"]
            last = messages[-1]
            if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
                return "tools"
            return "end"

        def consolidate_node(state):
            """Collect last tool outputs for the current step and advance index."""
            idx = int(state.get("step_idx") or 0)
            plan = state.get("plan") or {"steps": []}
            steps: List[dict] = plan.get("steps", [])
            if idx >= len(steps):
                return {}
            sid = steps[idx].get("id", idx + 1)
            # Gather tool messages with matching id prefix
            contents: List[str] = []
            for msg in state["messages"]:
                if isinstance(msg, ToolMessage):
                    tcid = getattr(msg, "tool_call_id", "")
                    if tcid == f"pe{sid}":
                        contents.append(msg.content if isinstance(msg.content, str) else str(msg.content))
            # Update results list
            results = list(state.get("results") or [])
            results.append({"step_id": sid, "output": "\n".join(contents)})
            return {"results": results, "step_idx": idx + 1}

        def synthesizer_node(state):
            """Synthesize a final answer from the collected step results."""
            messages = state["messages"]
            plan = state.get("plan") or {"steps": []}
            steps: List[dict] = plan.get("steps", [])
            results = list(state.get("results") or [])

            # Build a compact summary for the LLM to synthesize from
            lines = []
            for s in steps:
                sid = s.get("id")
                desc = s.get("description", "")
                out = next((r.get("output") for r in results if r.get("step_id") == sid), "")
                lines.append(f"Step {sid}: {desc}\nResult: {out}")
            synthesis_prompt = (
                "Based on the following executed plan steps and their results, provide a complete answer to the user's task.\n\n"
                + "\n\n".join(lines)
            )
            response = llm_with_tools.invoke(messages + [SystemMessage(content=synthesis_prompt)])
            return {"messages": [response]}

        # Create the graph
        workflow = StateGraph(agent_state_class)

        # Add nodes
        workflow.add_node("planner", planner_node)
        workflow.add_node("executor", step_to_calls_node)
        workflow.add_node("tools", ToolNode(tools))
        workflow.add_node("consolidate", consolidate_node)
        workflow.add_node("synthesizer", synthesizer_node)

        # Define the flow
        workflow.add_edge(START, "planner")

        # After planning, start executing if there are steps
        workflow.add_conditional_edges(
            "planner",
            after_planner_route,
            {
                "execute": "executor",
                "end": END,
            }
        )

        # After emitting tool call, go to tools
        workflow.add_conditional_edges(
            "executor",
            after_execute_route,
            {
                "tools": "tools",
                "end": END,
            }
        )

        # After tools, consolidate results for this step
        workflow.add_edge("tools", "consolidate")
        # After consolidation, either execute next step or end
        def after_consolidate_route(state) -> Literal["execute", "synthesize"]:
            plan = state.get("plan") or {"steps": []}
            steps = plan.get("steps", [])
            idx = int(state.get("step_idx") or 0)
            return "execute" if idx < len(steps) else "synthesize"
        workflow.add_conditional_edges(
            "consolidate",
            after_consolidate_route,
            {
                "execute": "executor",
                "synthesize": "synthesizer",
            }
        )

        # After synthesis, end
        workflow.add_edge("synthesizer", END)

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
