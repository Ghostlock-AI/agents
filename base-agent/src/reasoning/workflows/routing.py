"""
Routing Workflow Pattern

Classifies inputs and directs them to specialized downstream handlers.
This enables separation of concerns and prevents single optimization from
degrading performance across diverse input types.

Benefits:
- Efficient resource usage (route simple queries to smaller models)
- Specialized handling per input category
- Clearer separation of concerns
- Better performance through focused optimization

Use cases:
- Customer service (general/refund/technical queries)
- Content moderation (safe/unsafe/needs-review)
- Cost optimization (simple → small model, complex → large model)
- Multi-domain assistants (coding/writing/research)

Pattern: Anthropic's "Routing" workflow from Building Effective Agents
"""

from typing import Literal, Dict, Callable, Any, Optional
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field

from reasoning.base import ReasoningStrategy


class RouteClassification(BaseModel):
    """Classification result for routing."""
    category: str = Field(description="The identified category for this input")
    confidence: float = Field(description="Confidence score 0-1", ge=0, le=1)
    reasoning: str = Field(description="Brief explanation for the classification")


class RoutingWorkflow(ReasoningStrategy):
    """
    Routing workflow implementation.

    Routes inputs to specialized handlers based on classification.
    """

    def __init__(
        self,
        categories: Dict[str, str],
        handlers: Optional[Dict[str, Callable]] = None,
        default_category: str = "general",
    ):
        """
        Initialize routing workflow.

        Args:
            categories: Dict mapping category names to descriptions
                       e.g., {"technical": "Technical support questions",
                              "billing": "Billing and payment issues",
                              "general": "General inquiries"}
            handlers: Optional dict mapping categories to handler functions
                     If not provided, uses default LLM handler for all
            default_category: Fallback category for unclear inputs
        """
        self.categories = categories
        self.handlers = handlers or {}
        self.default_category = default_category

    def get_name(self) -> str:
        return "routing"

    def get_description(self) -> str:
        return (
            "Routing: Classifies inputs and routes to specialized handlers. "
            "Best for: diverse input types, cost optimization, domain separation."
        )

    def get_config(self) -> dict:
        return {
            "categories": list(self.categories.keys()),
            "default_category": self.default_category,
            "num_handlers": len(self.handlers),
        }

    def create_graph(self, agent_state_class, llm_with_tools, tools):
        """Create the routing workflow graph."""

        # Build classification prompt
        categories_list = "\n".join(
            [f"- {name}: {desc}" for name, desc in self.categories.items()]
        )

        classification_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                f"""You are a classification expert. Analyze the user's input and classify it into ONE of these categories:

{categories_list}

Return ONLY a JSON object with this structure:
{{
    "category": "category_name",
    "confidence": 0.95,
    "reasoning": "brief explanation"
}}

Choose the most appropriate category. If uncertain, use "{self.default_category}".
""",
            ),
            ("human", "{input}")
        ])

        def classifier_node(state):
            """Classify the input into a category."""
            messages = state["messages"]

            # Get the last human message
            user_input = None
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage):
                    user_input = msg.content
                    break

            if not user_input:
                user_input = str(messages[-1].content)

            # Use structured output for classification
            llm_with_structure = llm_with_tools.with_structured_output(RouteClassification)
            prompt = classification_prompt.format_messages(input=user_input)

            try:
                classification = llm_with_structure.invoke(prompt)
            except Exception:
                # Fallback to default category on error
                classification = RouteClassification(
                    category=self.default_category,
                    confidence=0.5,
                    reasoning="Classification failed, using default category"
                )

            # Store classification in state
            return {
                "messages": [AIMessage(content=f"[CLASSIFIED: {classification.category}]")],
                "plan": {
                    "category": classification.category,
                    "confidence": classification.confidence,
                    "reasoning": classification.reasoning,
                }
            }

        def route_handler(state):
            """Route to the appropriate handler based on classification."""
            category = state.get("plan", {}).get("category", self.default_category)

            # Get handler for this category or use default
            handler = self.handlers.get(category)

            if handler:
                # Use custom handler
                try:
                    result = handler(state)
                    return {"messages": [AIMessage(content=result)]}
                except Exception as e:
                    return {"messages": [AIMessage(content=f"Handler error: {str(e)}")]}
            else:
                # Use default LLM handler
                messages = state["messages"]

                # Build a specialized system prompt for this category
                category_desc = self.categories.get(category, "general inquiries")
                system_msg = SystemMessage(
                    content=f"You are a specialist in {category_desc}. "
                            f"Provide helpful, accurate responses in this domain."
                )

                # Invoke LLM with category-specific context
                response = llm_with_tools.invoke([system_msg] + messages)
                return {"messages": [response]}

        # Create the graph
        workflow = StateGraph(agent_state_class)

        # Add nodes
        workflow.add_node("classifier", classifier_node)
        workflow.add_node("handler", route_handler)

        # Define the flow
        workflow.add_edge(START, "classifier")
        workflow.add_edge("classifier", "handler")
        workflow.add_edge("handler", END)

        # Add memory
        memory = MemorySaver()

        return workflow.compile(checkpointer=memory)

    def get_trace_info(self, state=None) -> dict:
        """Get trace information."""
        info = super().get_trace_info(state)

        if state and "plan" in state:
            info["classification"] = state["plan"]

        return info
