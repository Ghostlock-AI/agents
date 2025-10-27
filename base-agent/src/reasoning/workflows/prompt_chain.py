"""
Prompt Chaining Workflow Pattern

Decomposes complex tasks into sequential steps where each LLM call processes
previous outputs. Includes programmatic "gates" to verify mid-process correctness.

Benefits:
- Simplifies individual LLM tasks for better accuracy
- Validates intermediate results with quality gates
- Clear progress tracking through pipeline
- Can short-circuit on validation failures

Use cases:
- Content generation → validation → translation
- Data extraction → formatting → verification
- Research → summarization → fact-checking
- Code generation → testing → documentation

Trade-off: Sacrifices latency for accuracy by breaking tasks into focused steps.

Pattern: Anthropic's "Prompt Chaining" workflow from Building Effective Agents
"""

from typing import List, Callable, Optional, Dict, Any
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from reasoning.base import ReasoningStrategy


class ChainStep(BaseModel):
    """Definition of a step in the prompt chain."""
    name: str = Field(description="Name of this step")
    prompt_template: str = Field(description="Prompt template for this step")
    validator: Optional[Callable[[str], tuple[bool, str]]] = Field(
        default=None,
        description="Optional validation function: (output) -> (is_valid, feedback)"
    )

    class Config:
        arbitrary_types_allowed = True


class PromptChainWorkflow(ReasoningStrategy):
    """
    Prompt chaining workflow implementation.

    Executes a sequence of LLM calls with optional validation gates.
    """

    def __init__(
        self,
        steps: Optional[List[ChainStep]] = None,
        stop_on_validation_failure: bool = True,
    ):
        """
        Initialize prompt chain workflow.

        Args:
            steps: List of chain steps to execute in order
                  If None, uses a default multi-step workflow
            stop_on_validation_failure: If True, halt chain on validation failure
                                       If False, continue with feedback
        """
        self.steps = steps or self._default_steps()
        self.stop_on_validation_failure = stop_on_validation_failure

    def _default_steps(self) -> List[ChainStep]:
        """Default 3-step chain: Generate → Validate → Refine."""
        return [
            ChainStep(
                name="generate",
                prompt_template="Generate a response to the user's request: {input}",
            ),
            ChainStep(
                name="validate",
                prompt_template=(
                    "Review this response for accuracy and completeness:\n\n{previous}\n\n"
                    "Provide specific feedback on what could be improved."
                ),
            ),
            ChainStep(
                name="refine",
                prompt_template=(
                    "Original request: {input}\n\n"
                    "Draft response: {step_0}\n\n"
                    "Feedback: {step_1}\n\n"
                    "Create an improved final response incorporating the feedback."
                ),
            ),
        ]

    def get_name(self) -> str:
        return "prompt-chain"

    def get_description(self) -> str:
        return (
            "Prompt Chain: Sequential LLM calls with validation gates. "
            "Best for: content pipelines, quality-critical tasks, structured workflows."
        )

    def get_config(self) -> dict:
        return {
            "num_steps": len(self.steps),
            "steps": [step.name for step in self.steps],
            "stop_on_failure": self.stop_on_validation_failure,
        }

    def create_graph(self, agent_state_class, llm_with_tools, tools):
        """Create the prompt chain workflow graph."""

        def execute_step(step_index: int):
            """Create a node function for executing a specific step."""
            step = self.steps[step_index]

            def step_node(state):
                messages = state["messages"]
                results = state.get("results", [])

                # Get the original user input
                user_input = None
                for msg in reversed(messages):
                    if isinstance(msg, HumanMessage):
                        user_input = msg.content
                        break

                if not user_input:
                    user_input = str(messages[-1].content)

                # Build context with previous step results
                context = {"input": user_input}

                # Add previous step results
                for i, result in enumerate(results):
                    context[f"step_{i}"] = result.get("output", "")

                # Get the most recent output for "previous" placeholder
                if results:
                    context["previous"] = results[-1].get("output", "")

                # Format the prompt with context
                try:
                    prompt_text = step.prompt_template.format(**context)
                except KeyError as e:
                    # Handle missing placeholder gracefully
                    prompt_text = step.prompt_template

                # Invoke LLM
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a helpful assistant executing a multi-step workflow."),
                    ("human", prompt_text)
                ])

                llm_without_tools = llm_with_tools.bind(tools=[])
                response = llm_without_tools.invoke(prompt.format_messages())

                output = response.content if hasattr(response, 'content') else str(response)

                # Validate if validator provided
                is_valid = True
                feedback = ""

                if step.validator:
                    try:
                        is_valid, feedback = step.validator(output)
                    except Exception as e:
                        is_valid = False
                        feedback = f"Validation error: {str(e)}"

                # Store result
                step_result = {
                    "step_name": step.name,
                    "output": output,
                    "is_valid": is_valid,
                    "feedback": feedback,
                }

                new_results = results + [step_result]

                # Update state
                return {
                    "messages": [AIMessage(content=f"[{step.name.upper()}] {output[:100]}...")],
                    "results": new_results,
                    "plan": {"current_step": step_index, "is_valid": is_valid},
                }

            return step_node

        def route_after_step(state) -> str:
            """Determine next node after a step."""
            plan = state.get("plan", {})
            current_step = plan.get("current_step", 0)
            is_valid = plan.get("is_valid", True)

            # Check if validation failed and we should stop
            if not is_valid and self.stop_on_validation_failure:
                return "end"

            # Check if more steps remain
            if current_step + 1 < len(self.steps):
                return f"step_{current_step + 1}"

            return "end"

        def synthesizer_node(state):
            """Synthesize final output from all step results."""
            results = state.get("results", [])

            if not results:
                return {"messages": [AIMessage(content="No results to synthesize.")]}

            # Use the last step's output as final result
            final_output = results[-1].get("output", "")

            # Check if any validations failed
            failed_validations = [
                r for r in results
                if not r.get("is_valid", True)
            ]

            if failed_validations:
                feedback_summary = "\n".join([
                    f"- {r['step_name']}: {r['feedback']}"
                    for r in failed_validations
                ])
                final_output = (
                    f"{final_output}\n\n"
                    f"Note: Some validation checks failed:\n{feedback_summary}"
                )

            return {"messages": [AIMessage(content=final_output)]}

        # Create the graph
        workflow = StateGraph(agent_state_class)

        # Add step nodes
        for i in range(len(self.steps)):
            workflow.add_node(f"step_{i}", execute_step(i))

        # Add synthesizer
        workflow.add_node("synthesizer", synthesizer_node)

        # Connect START to first step
        workflow.add_edge(START, "step_0")

        # Connect steps with conditional routing
        for i in range(len(self.steps)):
            if i < len(self.steps) - 1:
                # Conditional edge to next step or end
                workflow.add_conditional_edges(
                    f"step_{i}",
                    route_after_step,
                    {
                        f"step_{i + 1}": f"step_{i + 1}",
                        "end": "synthesizer",
                    }
                )
            else:
                # Last step always goes to synthesizer
                workflow.add_edge(f"step_{i}", "synthesizer")

        # Synthesizer to END
        workflow.add_edge("synthesizer", END)

        # Add memory
        memory = MemorySaver()

        return workflow.compile(checkpointer=memory)

    def get_trace_info(self, state=None) -> dict:
        """Get trace information."""
        info = super().get_trace_info(state)

        if state:
            results = state.get("results", [])
            info["completed_steps"] = [r["step_name"] for r in results]
            info["validation_status"] = [
                {"step": r["step_name"], "valid": r.get("is_valid", True)}
                for r in results
            ]

        return info
