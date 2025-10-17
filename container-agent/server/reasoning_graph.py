"""
Reasoning graph with Router → Planner → Agent ↔ Tools ↔ Reflect cycle.
Search-first policy when needed and shell for file/code operations.
"""

from __future__ import annotations

from typing import Annotated, List, Literal, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from server.tools import get_tools


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    goal: str
    plan: List[str]
    cursor: int
    evidence: List[str]
    done: bool


SYSTEM_PROMPT = (
    "You are a research and code agent that follows a ReAct loop.\n"
    "Tools available: search (if configured), web_fetch(url), shell.\n"
    "Policy: Prefer search for novel/uncertain facts, then web_fetch 1–2 sources and cite.\n"
    "Use shell only for local file/code operations and simple commands.\n"
    "Answer clearly, cite sources when based on the web, and keep shell use minimal and safe.\n"
)


def _needs_search(text: str) -> bool:
    t = text.lower()
    triggers = (
        "latest",
        "today",
        "current",
        "news",
        "source",
        "cite",
        "citation",
        "compare",
        "vs",
        "who is",
        "what is",
        "according to",
        "update",
    )
    return any(k in t for k in triggers)


def _make_initial_plan(goal: str, needs_search: bool) -> List[str]:
    plan: List[str] = []
    if needs_search:
        plan += [
            "search for relevant sources",
            "web_fetch 1–2 promising links",
            "synthesize findings with citations",
        ]
    else:
        plan += ["reason and answer"]
    # If the user asks to create or modify files, include a shell step.
    gl = goal.lower()
    if any(k in gl for k in ("write file", "create file", "edit", "script", "code")):
        plan.append("use shell to write/edit files and validate")
    return plan


def build_reasoning_graph(llm: ChatOpenAI):
    tools = get_tools()
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    builder = StateGraph(AgentState)

    def router(state: AgentState) -> dict:
        # Derive goal from last user message if absent
        goal = state.get("goal") or next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            "",
        )
        needs_search = _needs_search(goal)
        plan = state.get("plan") or _make_initial_plan(goal, needs_search)
        return {"goal": goal, "plan": plan, "cursor": 0, "done": False, "evidence": state.get("evidence", [])}

    def planner(state: AgentState) -> dict:
        # Simple no-op planner if plan already exists; could be expanded to replan
        plan = state.get("plan") or _make_initial_plan(state.get("goal", ""), _needs_search(state.get("goal", "")))
        return {"plan": plan}

    def agent(state: AgentState) -> dict:
        messages = state["messages"]
        # Ensure a single system message at the start encapsulating policy and current plan
        if not any(isinstance(m, SystemMessage) for m in messages):
            plan_txt = "\n".join(f"- {s}" for s in state.get("plan", []))
            sys = SystemMessage(
                content=(
                    f"{SYSTEM_PROMPT}\n\nCurrent goal: {state.get('goal','')}\n"
                    f"Planned steps:\n{plan_txt if plan_txt else '- (none)'}\n"
                )
            )
            messages = [sys] + messages

        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def route_from_agent(state: AgentState) -> Literal["tools", "end"]:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
            return "tools"
        return "end"

    # Wire nodes
    builder.add_node("router", router)
    builder.add_node("planner", planner)
    builder.add_node("agent", agent)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "router")
    builder.add_edge("router", "planner")
    builder.add_edge("planner", "agent")
    builder.add_conditional_edges("agent", route_from_agent, {"tools": "tools", "end": END})
    builder.add_edge("tools", "agent")

    return builder.compile(checkpointer=MemorySaver())
