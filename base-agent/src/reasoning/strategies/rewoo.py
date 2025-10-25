"""
ReWOO (Reasoning WithOut Observation) Strategy

Advanced planning strategy that:
1. Plans ALL steps upfront (without executing)
2. Executes all tool calls in parallel
3. Synthesizes final answer from all results

Benefits:
- Massive speedup via parallel execution
- Better for predictable/structured tasks
- Reduces token usage (fewer LLM calls)

Drawbacks:
- Can't adapt plan based on tool results
- Requires clear problem structure upfront
- Less effective for exploratory tasks

Best for:
- Research tasks (multiple searches in parallel)
- Data gathering from multiple sources
- Tasks with clear, predictable steps

Research: https://arxiv.org/abs/2305.18323
"""

from typing import List, Dict, Any, Literal, Optional, Set
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from .base import ReasoningStrategy
from reasoning.tool_context import build_tool_guide


class Plan(BaseModel):
    """A plan with multiple steps to execute."""
    steps: List[Dict[str, Any]] = Field(
        description="List of steps, each with 'tool', 'args', and 'depends_on' keys"
    )


class ReWOOStrategy(ReasoningStrategy):
    """ReWOO (Reasoning Without Observation) strategy implementation."""

    def __init__(self):
        """Initialize ReWOO strategy."""
        self._plan = None
        self._execution_results = {}

    def get_name(self) -> str:
        return "rewoo"

    def get_description(self) -> str:
        return (
            "ReWOO (Reasoning Without Observation): Plans all steps upfront, "
            "then executes tools in parallel for maximum speed. "
            "Best for: research tasks, data gathering, predictable workflows."
        )

    def create_graph(self, agent_state_class, llm_with_tools, tools):
        """Create the ReWOO reasoning graph with explicit execution of planned steps."""

        # Create a planning prompt that demands strict JSON with exact tool names
        planning_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are a strategic planner. Given a user query, return a STRICT JSON plan with steps.

TOOL CONTEXT (catalog, rules, examples):
{tool_guide}

JSON schema (example):
{{
  "steps": [
    {{"id": 1, "tool": "ddgs_search", "args": {{"query": "...", "max_results": 5}}, "depends_on": []}},
    {{"id": 2, "tool": "web_fetch",   "args": {{"url": "..."}},            "depends_on": [1]}}
  ]
}}

Rules:
- Use integers for step ids starting at 1.
- Use depends_on to express dependencies by id.
- Ensure arguments match the tool signatures.
- Do NOT include any text outside JSON.
""",
            ),
            ("human", "{query}")
        ])

        def _tools_signature() -> str:
            sigs = []
            for t in tools:
                name = getattr(t, "name", "")
                doc = getattr(t, "description", "")
                sigs.append(f"- {name}: {doc}")
            return "\n".join(sigs)

        def _parse_json(text: str) -> Optional[Dict[str, Any]]:
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
            """Create a plan for solving the task and store it in state['plan']."""
            messages = state["messages"]
            query = None
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage):
                    query = msg.content
                    break
            if not query:
                query = str(messages[-1].content)

            prompt = planning_prompt.format_messages(
                tool_guide=build_tool_guide(tools),
                query=query,
            )
            response = llm_with_tools.invoke(prompt)
            text = response.content if isinstance(response.content, str) else str(response.content)
            plan_json = _parse_json(text)
            if not plan_json or not isinstance(plan_json, dict) or "steps" not in plan_json:
                plan_json = {"steps": [
                    {"id": 1, "tool": "ddgs_search", "args": {"query": query, "max_results": 5}, "depends_on": []}
                ]}
            return {
                "messages": [AIMessage(content=f"[PLAN CREATED]\n\n{text}")],
                "plan": plan_json,
                "executed": [],
                "results": [],
            }

        def plan_to_calls_node(state):
            """Expand ready plan steps into explicit tool calls (can be parallel)."""
            plan = state.get("plan") or {}
            steps: List[Dict[str, Any]] = plan.get("steps", [])
            executed: Set[str] = set(state.get("executed") or [])

            def is_ready(s: Dict[str, Any]) -> bool:
                sid = str(s.get("id"))
                if sid in executed:
                    return False
                deps = s.get("depends_on", []) or []
                return all(str(d) in executed for d in deps)

            ready = [s for s in steps if is_ready(s)]
            tool_calls = []
            for s in ready:
                tool_name = s.get("tool")
                args = s.get("args") or {}
                sid = str(s.get("id"))
                if tool_name:
                    tool_calls.append({"name": tool_name, "args": args, "id": f"s{sid}"})

            if not tool_calls:
                return {"messages": []}
            return {"messages": [AIMessage(content="", tool_calls=tool_calls)]}

        def collect_results_node(state):
            """Collect tool results and mark corresponding steps as executed."""
            messages = state["messages"]
            executed: Set[str] = set(state.get("executed") or [])
            results: List[Dict[str, Any]] = list(state.get("results") or [])

            for msg in messages:
                if isinstance(msg, ToolMessage):
                    tcid = getattr(msg, "tool_call_id", None)
                    name = getattr(msg, "name", None)
                    if tcid and tcid.startswith("s"):
                        sid = tcid[1:]
                        # Only record once per step id
                        if sid not in executed:
                            executed.add(sid)
                            results.append({
                                "step_id": sid,
                                "tool": name,
                                "content": msg.content if isinstance(msg.content, str) else str(msg.content),
                            })

            return {"executed": list(executed), "results": results}

        def synthesizer_node(state):
            """Synthesize final answer from collected execution results."""
            messages = state["messages"]
            results: List[Dict[str, Any]] = list(state.get("results") or [])
            lines = []
            for r in results:
                lines.append(f"- [{r.get('tool')}] step {r.get('step_id')}: {r.get('content')}")
            synthesis_prompt = (
                "Based on the following tool execution results, provide a concise, complete answer to the original query.\n\n"
                + "\n".join(lines)
            )
            response = llm_with_tools.invoke(messages + [SystemMessage(content=synthesis_prompt)])
            return {"messages": [response]}

        def planner_route(state) -> Literal["plan_to_calls", "synthesizer"]:
            plan = state.get("plan") or {}
            steps = plan.get("steps", [])
            return "plan_to_calls" if steps else "synthesizer"

        def after_plan_route(state) -> Literal["tools", "synthesizer"]:
            plan = state.get("plan") or {}
            steps: List[Dict[str, Any]] = plan.get("steps", [])
            executed: Set[str] = set(state.get("executed") or [])
            for s in steps:
                sid = str(s.get("id"))
                if sid in executed:
                    continue
                deps = s.get("depends_on", []) or []
                if all(str(d) in executed for d in deps):
                    return "tools"
            return "synthesizer"

        def after_tools_route(state) -> Literal["plan_to_calls", "synthesizer"]:
            plan = state.get("plan") or {}
            steps: List[Dict[str, Any]] = plan.get("steps", [])
            executed: Set[str] = set(state.get("executed") or [])
            for s in steps:
                if str(s.get("id")) not in executed:
                    return "plan_to_calls"
            return "synthesizer"

        # Create the graph
        workflow = StateGraph(agent_state_class)

        # Add nodes
        workflow.add_node("planner", planner_node)
        workflow.add_node("plan_to_calls", plan_to_calls_node)
        workflow.add_node("tools", ToolNode(tools))
        workflow.add_node("collect", collect_results_node)
        workflow.add_node("synthesizer", synthesizer_node)

        # Define the flow
        workflow.add_edge(START, "planner")

        # After planning, expand to calls or synthesize
        workflow.add_conditional_edges(
            "planner",
            planner_route,
            {
                "plan_to_calls": "plan_to_calls",
                "synthesizer": "synthesizer",
            }
        )

        # After expansion, either run tools or synthesize
        workflow.add_conditional_edges(
            "plan_to_calls",
            after_plan_route,
            {
                "tools": "tools",
                "synthesizer": "synthesizer",
            }
        )

        # After tools, collect then loop/finish
        workflow.add_edge("tools", "collect")
        workflow.add_conditional_edges(
            "collect",
            after_tools_route,
            {
                "plan_to_calls": "plan_to_calls",
                "synthesizer": "synthesizer",
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

        if self._plan:
            info["plan"] = self._plan

        if self._execution_results:
            info["execution_results"] = self._execution_results

        return info
