"""
Agent - Core agent class with reasoning loop and memory.
"""

import os
from typing import TypedDict, Annotated, Optional, Dict, Any, List
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages

from tools import get_tools
from reasoning.tool_context import build_tool_guide
from reasoning import get_global_registry, create_react_graph
from workflow_system import Workflow, WorkflowBuilder, WorkflowTemplates


class AgentState(TypedDict):
    """State that tracks conversation messages and optional planning state."""
    messages: Annotated[list[BaseMessage], add_messages]
    # Optional planning state used by some strategies (ReWOO / Plan-Execute)
    plan: Optional[Dict[str, Any]]  # Strategy-specific plan structure
    step_idx: Optional[int]         # Current step index for sequential execution
    executed: Optional[List[str]]   # Executed step ids (as strings)
    results: Optional[List[Dict[str, Any]]]  # Collected tool results / summaries


class Agent:
    """Base agent with LLM, reasoning strategies, tools, and memory."""

    def __init__(
        self,
        model_name: str = None,
        temperature: float = 0.7,
        system_prompt_path: str = "system_prompt.txt",
        tools: list = None,
        reasoning_strategy: str = "react",
        mode: str = "agent"  # "agent" or "workflow"
    ):
        """Initialize the agent.

        Args:
            model_name: OpenAI model name (default: gpt-4o-mini)
            temperature: LLM temperature (0.0-1.0)
            system_prompt_path: Path to system prompt file
            tools: List of tools available to agent
            reasoning_strategy: For agent mode: 'react', 'rewoo', 'plan-execute', 'lats'
            mode: "agent" for autonomous reasoning, "workflow" for sequential pipelines
        """
        self.model_name = model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.temperature = temperature
        self.system_prompt = self._load_system_prompt(system_prompt_path)
        self.mode = mode
        self.current_workflow = None

        # Initialize tools after env is loaded
        self.tools = tools or get_tools()

        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=self.temperature,
            streaming=True
        )

        # Bind tools to LLM
        # This tells the LLM about available tools and their descriptions
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # Initialize strategy registry (for agent mode)
        self.strategy_registry = get_global_registry()

        # Set initial reasoning strategy (only used in agent mode)
        if reasoning_strategy:
            self.strategy_registry.set_current_strategy(reasoning_strategy)

        # Build the reasoning graph using current strategy (only for agent mode)
        if self.mode == "agent":
            self.graph = self._build_graph()
        else:
            self.graph = None

    def _load_system_prompt(self, path: str) -> str:
        """Load system prompt from file."""
        prompt_path = Path(__file__).parent / path

        if prompt_path.exists():
            return prompt_path.read_text().strip()

        return "You are a helpful AI assistant."

    def _build_graph(self):
        """Build the reasoning graph using the current strategy."""
        strategy = self.strategy_registry.get_current_strategy()
        return strategy.create_graph(AgentState, self.llm_with_tools, self.tools)

    def switch_reasoning_strategy(self, strategy_name: str):
        """
        Switch to a different reasoning strategy.

        Args:
            strategy_name: Name of the strategy to switch to (e.g., 'react', 'rewoo', 'plan-execute', 'lats')

        Raises:
            KeyError: If strategy doesn't exist
        """
        self.strategy_registry.set_current_strategy(strategy_name)
        # Rebuild the graph with the new strategy
        self.graph = self._build_graph()

    def get_current_strategy_name(self) -> str:
        """Get the name of the currently active reasoning strategy."""
        return self.strategy_registry.get_current_strategy_name()

    def list_strategies(self) -> list:
        """List all available reasoning strategies with descriptions."""
        return self.strategy_registry.list_strategies()

    def get_strategy_info(self, strategy_name: str = None) -> dict:
        """
        Get detailed information about a strategy.

        Args:
            strategy_name: Name of the strategy (defaults to current strategy)

        Returns:
            Dict with strategy details including config
        """
        if strategy_name is None:
            strategy_name = self.get_current_strategy_name()
        return self.strategy_registry.get_strategy_info(strategy_name)

    # Workflow mode methods
    def set_workflow(self, workflow: Workflow):
        """
        Set a workflow for workflow mode.

        Args:
            workflow: Workflow instance to use

        Example:
            workflow = WorkflowBuilder("my_workflow")
                .add_step("research", "Gather information")
                .add_step("analyze", "Analyze findings")
                .build()
            agent.set_workflow(workflow)
        """
        if self.mode != "workflow":
            raise ValueError("Agent must be in workflow mode to set a workflow. Initialize with mode='workflow'")
        self.current_workflow = workflow

    def load_workflow_template(self, template_name: str):
        """
        Load a pre-built workflow template.

        Args:
            template_name: Name of template ('research_and_summarize', 'code_review')

        Example:
            agent = Agent(mode='workflow')
            agent.load_workflow_template('research_and_summarize')
        """
        if self.mode != "workflow":
            raise ValueError("Agent must be in workflow mode to load templates. Initialize with mode='workflow'")

        templates = {
            'research_and_summarize': WorkflowTemplates.research_and_summarize,
            'code_review': WorkflowTemplates.code_review,
        }

        if template_name not in templates:
            raise ValueError(f"Unknown template: {template_name}. Available: {list(templates.keys())}")

        self.current_workflow = templates[template_name]()

    def get_workflow_trace(self) -> list:
        """
        Get the execution trace from the last workflow run.

        Returns:
            List of step execution details
        """
        if not self.current_workflow:
            return []
        return self.current_workflow.get_execution_trace()

    def stream(self, user_input: str, thread_id: str = "default", show_trace: bool = False):
        """
        Stream response for user input.

        Args:
            user_input: The user's query
            thread_id: Session thread identifier
            show_trace: Whether to show reasoning trace information
        """
        from langchain_core.messages import ToolMessage

        # Handle workflow mode
        if self.mode == "workflow":
            if not self.current_workflow:
                yield "[ERROR: No workflow set. Use set_workflow() or load_workflow_template()]\n"
                return

            yield f"[WORKFLOW: {self.current_workflow.name}]\n"

            # Show steps if trace enabled
            if show_trace:
                yield f"[STEPS: {len(self.current_workflow.steps)}]\n"
                for i, step in enumerate(self.current_workflow.steps):
                    yield f"  Step {i+1}: {step.name} - {step.instruction[:50]}...\n"

            # Execute workflow
            try:
                # Show step execution in real-time
                for i, step in enumerate(self.current_workflow.steps):
                    yield f"[STEP {i+1}/{len(self.current_workflow.steps)}: {step.name}]\n"

                # Execute and get result
                result = self.current_workflow.execute(user_input, self.llm, self.tools)
                yield result

                # Show trace if requested
                if show_trace:
                    yield "\n[WORKFLOW TRACE]\n"
                    for log_entry in self.current_workflow.get_execution_trace():
                        yield f"  {log_entry.get('step')}: {log_entry.get('name')}\n"

            except Exception as e:
                yield f"[WORKFLOW ERROR: {e}]\n"
            return

        # Handle agent mode (existing logic)
        # Prepend system message to the user input
        input_state = {
            "messages": [
                SystemMessage(content=self.system_prompt),
                # Inject shared tool guide so all strategies see the same context
                SystemMessage(content=build_tool_guide(self.tools)),
                HumanMessage(content=user_input),
            ]
        }

        config = {
            "configurable": {
                "thread_id": thread_id
            }
        }

        # Get strategy trace info if requested
        if show_trace:
            strategy = self.strategy_registry.get_current_strategy()
            trace_info = strategy.get_trace_info()
            yield f"[TRACE: {trace_info.get('strategy', 'unknown').upper()}]\n"

        # Stream and yield AI message content
        for chunk in self.graph.stream(input_state, config, stream_mode="values"):
            if "messages" in chunk and chunk["messages"]:
                last_message = chunk["messages"][-1]

                # Check for tool calls in AI messages
                if isinstance(last_message, AIMessage):
                    # If the AI is calling tools, notify the user
                    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                        for tool_call in last_message.tool_calls:
                            tool_name = tool_call.get("name", "unknown")
                            yield f"[TOOL: {tool_name}]\n"

                    # Show strategy-specific markers for LATS and Plan-Execute
                    if last_message.content:
                        content = last_message.content
                        # Show trace markers for special reasoning steps
                        if "[CANDIDATES GENERATED]" in content or "[PLAN CREATED]" in content or "[REFLECTION]" in content:
                            yield content
                        # Only yield if there's actual content (not just tool calls)
                        elif content:
                            yield content

                # Show tool results
                elif isinstance(last_message, ToolMessage):
                    # Don't display tool output, just let it feed back to agent
                    pass
