"""
Workflow System - Sequential LLM Pipeline Builder

This system allows building workflows where:
1. Each step = 1 LLM call with a specific job
2. Output of step N → Input to step N+1
3. Final synthesizer creates user-facing response

Example:
    workflow = WorkflowBuilder()
    workflow.add_step("research", "Research the topic and gather key facts")
    workflow.add_step("analyze", "Analyze the facts and identify patterns")
    workflow.add_step("draft", "Write a clear summary")
    workflow.set_synthesizer("Polish the summary for the user")

    result = workflow.execute("Tell me about LangGraph", llm, tools)
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage


@dataclass
class WorkflowStep:
    """A single step in a workflow pipeline."""

    name: str
    instruction: str
    result: Optional[str] = None

    def execute(self, input_text: str, llm, tools: List = None) -> str:
        """Execute this step with the given input."""
        # Create a focused prompt for this step
        prompt = f"""You are performing step '{self.name}' in a workflow.

Your specific job: {self.instruction}

Input from previous step:
{input_text}

Provide your output for this step. Be concise and focused on your specific job.
Do NOT try to complete the entire workflow - just do your assigned task.
If your instruction mentions using a tool, you MUST call that tool to get actual data.
"""

        # Call LLM with tools if available
        if tools:
            llm_with_tools = llm.bind_tools(tools)
            response = llm_with_tools.invoke([HumanMessage(content=prompt)])

            # Check if LLM wants to call tools
            if hasattr(response, 'tool_calls') and response.tool_calls:
                # Execute tools and get results
                from langgraph.prebuilt import ToolNode
                tool_node = ToolNode(tools)

                # Create a messages list for tool execution
                messages = [HumanMessage(content=prompt), response]

                # Execute tools
                tool_results = tool_node.invoke({"messages": messages})

                # Get tool outputs
                tool_outputs = []
                for msg in tool_results.get("messages", []):
                    if hasattr(msg, 'content'):
                        tool_outputs.append(str(msg.content))

                # Combine tool outputs with original response
                self.result = "\n".join(tool_outputs) if tool_outputs else (response.content if hasattr(response, 'content') else str(response))
            else:
                self.result = response.content if hasattr(response, 'content') else str(response)
        else:
            response = llm.invoke([HumanMessage(content=prompt)])
            self.result = response.content if hasattr(response, 'content') else str(response)

        return self.result


@dataclass
class Workflow:
    """A complete workflow with multiple steps and a synthesizer."""

    name: str
    steps: List[WorkflowStep] = field(default_factory=list)
    synthesizer_instruction: str = "Synthesize the results into a clear, helpful response for the user."
    execution_log: List[Dict[str, Any]] = field(default_factory=list)

    def execute(self, user_input: str, llm, tools: List = None) -> str:
        """
        Execute the workflow pipeline.

        Args:
            user_input: The user's original query
            llm: The LLM instance to use
            tools: Optional tools available to steps

        Returns:
            Final synthesized output
        """
        self.execution_log = []

        # Start with user input
        current_input = f"Original query: {user_input}"

        # Execute each step sequentially
        for i, step in enumerate(self.steps):
            self.execution_log.append({
                "step": i + 1,
                "name": step.name,
                "instruction": step.instruction,
                "input": current_input[:200] + "..." if len(current_input) > 200 else current_input,
            })

            # Execute step
            output = step.execute(current_input, llm, tools)

            self.execution_log[-1]["output"] = output[:200] + "..." if len(output) > 200 else output

            # Output becomes next input
            current_input = f"Previous step ({step.name}): {output}"

        # Final synthesis
        synthesis_prompt = f"""{self.synthesizer_instruction}

Original user query:
{user_input}

Workflow execution results:
"""

        for i, step in enumerate(self.steps):
            synthesis_prompt += f"\nStep {i+1} ({step.name}):\n{step.result}\n"

        synthesis_prompt += "\nProvide a final, polished response to the user's original query."

        # Call synthesizer (no tools, just text processing)
        response = llm.invoke([HumanMessage(content=synthesis_prompt)])
        final_output = response.content if hasattr(response, 'content') else str(response)

        self.execution_log.append({
            "step": "synthesizer",
            "name": "Final Synthesis",
            "output": final_output[:200] + "..." if len(final_output) > 200 else final_output,
        })

        return final_output

    def get_execution_trace(self) -> List[Dict[str, Any]]:
        """Get the execution trace for debugging/transparency."""
        return self.execution_log


class WorkflowBuilder:
    """
    Builder for creating workflows.

    Example:
        builder = WorkflowBuilder("research_workflow")
        builder.add_step("gather", "Collect relevant information")
        builder.add_step("analyze", "Analyze the information")
        builder.set_synthesizer("Create final summary")
        workflow = builder.build()
    """

    def __init__(self, name: str = "unnamed_workflow"):
        self.name = name
        self.steps: List[WorkflowStep] = []
        self.synthesizer_instruction = "Synthesize the results into a clear, helpful response."

    def add_step(self, name: str, instruction: str) -> 'WorkflowBuilder':
        """
        Add a step to the workflow.

        Args:
            name: Short name for this step (e.g., "research", "analyze")
            instruction: What this LLM call should accomplish

        Returns:
            self for chaining
        """
        self.steps.append(WorkflowStep(name=name, instruction=instruction))
        return self

    def set_synthesizer(self, instruction: str) -> 'WorkflowBuilder':
        """
        Set the instruction for the final synthesizer LLM call.

        Args:
            instruction: What the synthesizer should do with all step outputs

        Returns:
            self for chaining
        """
        self.synthesizer_instruction = instruction
        return self

    def build(self) -> Workflow:
        """Build and return the workflow."""
        if not self.steps:
            raise ValueError("Workflow must have at least one step")

        return Workflow(
            name=self.name,
            steps=self.steps.copy(),
            synthesizer_instruction=self.synthesizer_instruction
        )


# Pre-built workflow templates
class WorkflowTemplates:
    """Common workflow templates users can start from."""

    @staticmethod
    def research_and_summarize() -> Workflow:
        """
        Template: Research a topic and create a summary.

        Steps:
        1. Research: Gather information using tools
        2. Analyze: Extract key insights
        3. Draft: Write initial summary
        Synthesize: Polish for user
        """
        return (WorkflowBuilder("research_and_summarize")
                .add_step("research", "Search for and gather relevant information about the topic. Use tools to find current, accurate data.")
                .add_step("analyze", "Analyze the gathered information and identify the most important points, patterns, and insights.")
                .add_step("draft", "Write a clear, well-organized summary of the topic based on your analysis.")
                .set_synthesizer("Polish the summary for clarity and completeness. Format it nicely for the user.")
                .build())

    @staticmethod
    def code_review() -> Workflow:
        """
        Template: Review code and provide feedback.

        Steps:
        1. Read: Understand the code structure
        2. Check: Identify issues (bugs, style, security)
        3. Suggest: Propose improvements
        Synthesize: Format review report
        """
        return (WorkflowBuilder("code_review")
                .add_step("understand", "Read and understand the code's purpose, structure, and logic.")
                .add_step("check", "Identify any issues: bugs, style violations, security concerns, performance problems.")
                .add_step("suggest", "Propose specific improvements and best practices.")
                .set_synthesizer("Create a well-formatted code review report with clear sections: Overview, Issues Found, Recommendations.")
                .build())

    @staticmethod
    def simple_chain(steps: List[tuple[str, str]]) -> Workflow:
        """
        Create a simple custom workflow from a list of (name, instruction) tuples.

        Args:
            steps: List of (step_name, step_instruction) tuples

        Returns:
            Workflow with the specified steps
        """
        builder = WorkflowBuilder("custom_chain")
        for name, instruction in steps:
            builder.add_step(name, instruction)
        return builder.build()
