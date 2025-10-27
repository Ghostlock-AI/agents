"""
LATS (Language Agent Tree Search) Strategy

Advanced reasoning using Monte Carlo Tree Search with LLM-powered evaluation:
1. Generate multiple candidate actions
2. Simulate/evaluate each action
3. Select best action based on value function
4. Repeat with self-reflection

Benefits:
- Explores multiple solution paths
- Self-corrects through reflection
- Best for complex problems with multiple approaches
- 94.4% on HumanEval (research benchmark)

Drawbacks:
- Much slower (many LLM calls)
- Higher token usage
- Overkill for simple tasks

Best for:
- Complex algorithmic problems
- Tasks with multiple valid approaches
- When quality matters more than speed
- Code generation & optimization

Research: https://arxiv.org/abs/2310.04406
Note: This is a simplified LATS implementation. Full MCTS with UCB selection
would require more infrastructure.
"""

from typing import List, Dict, Any, Literal
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from reasoning.base import ReasoningStrategy


class LATSStrategy(ReasoningStrategy):
    """Simplified LATS (Language Agent Tree Search) strategy."""

    def __init__(
        self,
        num_candidates: int = 3,
        max_depth: int = 5,
        enable_reflection: bool = True
    ):
        """
        Initialize LATS strategy.

        Args:
            num_candidates: Number of alternative actions to consider at each step
            max_depth: Maximum search depth
            enable_reflection: Whether to use self-reflection for evaluation
        """
        self.num_candidates = num_candidates
        self.max_depth = max_depth
        self.enable_reflection = enable_reflection
        self._current_depth = 0
        self._action_history = []

    def get_name(self) -> str:
        return "lats"

    def get_description(self) -> str:
        return (
            "LATS (Language Agent Tree Search): Explores multiple solution paths, "
            "evaluates them through self-reflection, and selects the best approach. "
            "Best for: complex problems, code generation, tasks requiring exploration. "
            "WARNING: Much slower and uses more tokens than other strategies."
        )

    def get_config(self) -> dict:
        return {
            "num_candidates": self.num_candidates,
            "max_depth": self.max_depth,
            "enable_reflection": self.enable_reflection,
            "current_depth": self._current_depth,
        }

    def update_config(self, **kwargs) -> None:
        if "num_candidates" in kwargs:
            self.num_candidates = kwargs["num_candidates"]
        if "max_depth" in kwargs:
            self.max_depth = kwargs["max_depth"]
        if "enable_reflection" in kwargs:
            self.enable_reflection = kwargs["enable_reflection"]

    def create_graph(self, agent_state_class, llm_with_tools, tools):
        """Create the LATS reasoning graph."""

        # Prompt for generating multiple candidate actions
        candidate_generation_prompt = """You are exploring multiple approaches to solve a problem.

Current task: {task}

Previous actions taken:
{history}

Generate {num_candidates} different candidate approaches to proceed. For each candidate:
1. Describe the approach
2. Explain why it might work
3. Identify potential risks

Be creative and consider diverse strategies. Number each candidate clearly (1, 2, 3, etc.)."""

        # Prompt for reflection and evaluation
        reflection_prompt = """You are evaluating different approaches to solve a problem.

Task: {task}

Candidate approaches:
{candidates}

For each candidate, evaluate:
1. Likelihood of success (0-10)
2. Potential issues or failure modes
3. Required resources/tools

Then select the BEST candidate and explain why. Format: "SELECTED: [number]" followed by reasoning."""

        def generate_candidates_node(state):
            """Generate multiple candidate actions."""
            messages = state["messages"]
            self._current_depth += 1

            # Get current context
            last_message = messages[-1]
            task = str(last_message.content)

            # Format history
            history = "\n".join([
                f"- {action}"
                for action in self._action_history
            ])

            # Generate candidates
            prompt_text = candidate_generation_prompt.format(
                task=task,
                history=history or "No actions taken yet",
                num_candidates=self.num_candidates
            )

            prompt_messages = messages + [SystemMessage(content=prompt_text)]
            response = llm_with_tools.invoke(prompt_messages)

            # Store candidates in a special message
            candidates_message = AIMessage(
                content=f"[CANDIDATES GENERATED]\n\n{response.content}"
            )

            return {"messages": [candidates_message]}

        def reflect_and_select_node(state):
            """Reflect on candidates and select best one."""
            messages = state["messages"]

            # Find the candidates message
            candidates_content = None
            task = None

            for msg in reversed(messages):
                if isinstance(msg, AIMessage) and "[CANDIDATES GENERATED]" in msg.content:
                    candidates_content = msg.content.replace("[CANDIDATES GENERATED]", "").strip()
                if isinstance(msg, HumanMessage):
                    task = msg.content

            if not candidates_content:
                # No candidates found, skip reflection
                return {"messages": []}

            # Create reflection prompt
            prompt_text = reflection_prompt.format(
                task=task or "Unknown task",
                candidates=candidates_content
            )

            reflection_messages = [SystemMessage(content=prompt_text)]
            response = llm_with_tools.invoke(reflection_messages)

            # Mark this as a reflection
            reflection_message = AIMessage(
                content=f"[REFLECTION]\n\n{response.content}"
            )

            return {"messages": [reflection_message]}

        def execute_action_node(state):
            """Execute the selected action."""
            messages = state["messages"]

            # Execute with tools available
            response = llm_with_tools.invoke(messages)

            # Log this action
            if not (hasattr(response, "tool_calls") and response.tool_calls):
                self._action_history.append(response.content[:100])

            return {"messages": [response]}

        def should_continue(state) -> Literal["candidates", "reflect", "execute", "tools", "end"]:
            """Determine next step in LATS process."""
            messages = state["messages"]
            last_message = messages[-1]

            # Check depth limit
            if self._current_depth >= self.max_depth:
                # Check if we have a final answer
                if not (hasattr(last_message, "tool_calls") and last_message.tool_calls):
                    return "end"

            # If we have tool calls, execute them
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return "tools"

            # If we just generated candidates, reflect on them
            if isinstance(last_message, AIMessage) and "[CANDIDATES GENERATED]" in last_message.content:
                if self.enable_reflection:
                    return "reflect"
                else:
                    return "execute"

            # If we just reflected, execute the selected action
            if isinstance(last_message, AIMessage) and "[REFLECTION]" in last_message.content:
                return "execute"

            # If we just executed, check if we need more exploration
            if self._current_depth < self.max_depth:
                # Decide if we should explore more or conclude
                content = str(last_message.content).lower()

                # If response seems incomplete or uncertain, explore more
                if any(word in content for word in ["however", "but", "alternatively", "maybe", "might"]):
                    return "candidates"

            # Default: end if we have a substantive response
            if last_message.content and len(str(last_message.content)) > 50:
                return "end"

            return "execute"

        # Create the graph
        workflow = StateGraph(agent_state_class)

        # Add nodes
        workflow.add_node("candidates", generate_candidates_node)
        workflow.add_node("reflect", reflect_and_select_node)
        workflow.add_node("execute", execute_action_node)
        workflow.add_node("tools", ToolNode(tools))

        # Define the flow
        workflow.add_edge(START, "candidates")

        # After generating candidates
        workflow.add_conditional_edges(
            "candidates",
            should_continue,
            {
                "candidates": "candidates",
                "reflect": "reflect",
                "execute": "execute",
                "tools": "tools",
                "end": END,
            }
        )

        # After reflection
        workflow.add_conditional_edges(
            "reflect",
            should_continue,
            {
                "candidates": "candidates",
                "reflect": "reflect",
                "execute": "execute",
                "tools": "tools",
                "end": END,
            }
        )

        # After execution
        workflow.add_conditional_edges(
            "execute",
            should_continue,
            {
                "candidates": "candidates",
                "reflect": "reflect",
                "execute": "execute",
                "tools": "tools",
                "end": END,
            }
        )

        # After tools, go back to execute
        workflow.add_edge("tools", "execute")

        # Add memory
        memory = MemorySaver()

        return workflow.compile(checkpointer=memory)

    def get_trace_info(self, state=None) -> dict:
        """Get trace information."""
        info = super().get_trace_info(state)
        info.update({
            "depth": self._current_depth,
            "max_depth": self.max_depth,
            "action_history": self._action_history,
            "num_candidates": self.num_candidates,
            "reflection_enabled": self.enable_reflection,
        })
        return info
