"""
Intelligent Pattern Selector

Automatically selects the best reasoning pattern (workflow vs agent) based on
the user's query, following Anthropic's "Building Effective Agents" guidance.

Key Decision Criteria:
- Open-ended tasks with unpredictable steps → AGENTS
- Predefined tasks with clear structure → WORKFLOWS
- Quality-critical content → Prompt Chain
- Diverse input types → Routing
- Research and data gathering → ReWOO
- Complex multi-step projects → Plan-Execute
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate


class PatternRecommendation(BaseModel):
    """Recommendation for which pattern to use."""
    pattern_type: str = Field(description="Either 'agent' or 'workflow'")
    pattern_name: str = Field(description="Specific pattern name (e.g., 'react', 'rewoo', 'prompt-chain')")
    confidence: float = Field(description="Confidence score 0-1", ge=0, le=1)
    reasoning: str = Field(description="Explanation for why this pattern was chosen")
    characteristics: Dict[str, bool] = Field(description="Task characteristics detected")


class PatternSelector:
    """
    Intelligent selector that recommends the best reasoning pattern.

    Based on Anthropic's guidance:
    - Workflows for predictable tasks with clear steps
    - Agents for open-ended problems requiring exploration
    """

    def __init__(self, llm):
        """Initialize with an LLM for classification."""
        self.llm = llm
        self._create_classification_prompt()

    def _create_classification_prompt(self):
        """Create the classification prompt based on Anthropic's guidance."""
        self.classification_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                """You are an expert at analyzing tasks and selecting the optimal reasoning pattern.

Based on Anthropic's "Building Effective Agents" framework, analyze the user's query and recommend the best pattern.

## Decision Framework

### AGENTS - Use when task is:
- **Open-ended**: Steps cannot be predetermined
- **Exploratory**: Requires trial-and-error or investigation
- **Dynamic**: Next steps depend on previous results
- **Unpredictable**: Solution path is not clear upfront

**Available Agents:**
- **react**: Iterative reasoning with tool use (general purpose, default agent)
- **lats**: Tree search with self-reflection (complex problems, slower but higher quality)

### WORKFLOWS - Use when task is:
- **Structured**: Clear sequence of steps
- **Predictable**: Steps can be planned upfront
- **Quality-critical**: Needs validation gates
- **Multi-domain**: Different input types need specialized handling
- **Parallel**: Independent subtasks can run simultaneously

**Available Workflows:**
- **prompt-chain**: Sequential LLM calls with validation (content generation, quality-critical)
- **routing**: Classify and route to handlers (customer service, multi-domain queries)
- **rewoo**: Plan all steps, execute in parallel (research, data gathering)
- **plan-execute**: Adaptive planning with replanning (complex projects, multi-step tasks)

## Task Characteristics to Detect

Analyze for these characteristics:
- **open_ended**: Can steps be predetermined? (yes → workflow, no → agent)
- **needs_validation**: Requires quality gates? (yes → prompt-chain)
- **has_categories**: Multiple distinct input types? (yes → routing)
- **parallel_tasks**: Independent subtasks? (yes → rewoo)
- **needs_planning**: Complex multi-step project? (yes → plan-execute)
- **exploratory**: Requires investigation and adaptation? (yes → react)
- **very_complex**: Extremely hard problem? (yes → lats)

## Selection Priority

1. If needs_validation → **prompt-chain**
2. If has_categories → **routing**
3. If parallel_tasks → **rewoo**
4. If needs_planning AND predictable → **plan-execute**
5. If open_ended OR exploratory → **react** (default agent)
6. If very_complex AND open_ended → **lats**
7. Default → **react**

Respond with ONLY a JSON object:
{{
    "pattern_type": "agent" or "workflow",
    "pattern_name": "react" | "lats" | "prompt-chain" | "routing" | "rewoo" | "plan-execute",
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of why this pattern was chosen",
    "characteristics": {{
        "open_ended": true/false,
        "needs_validation": true/false,
        "has_categories": true/false,
        "parallel_tasks": true/false,
        "needs_planning": true/false,
        "exploratory": true/false,
        "very_complex": true/false
    }}
}}
""",
            ),
            ("human", "Task: {query}\n\nRecommend the best reasoning pattern for this task.")
        ])

    def select_pattern(self, query: str) -> PatternRecommendation:
        """
        Analyze a query and recommend the best pattern.

        Args:
            query: The user's query or task description

        Returns:
            PatternRecommendation with pattern selection and reasoning
        """
        # Use structured output for reliable classification
        llm_with_structure = self.llm.with_structured_output(PatternRecommendation)

        prompt = self.classification_prompt.format_messages(query=query)

        try:
            recommendation = llm_with_structure.invoke(prompt)
            return recommendation
        except Exception as e:
            # Fallback to react (most general-purpose) on error
            return PatternRecommendation(
                pattern_type="agent",
                pattern_name="react",
                confidence=0.5,
                reasoning=f"Classification failed ({str(e)}), using default ReAct agent",
                characteristics={
                    "open_ended": True,
                    "needs_validation": False,
                    "has_categories": False,
                    "parallel_tasks": False,
                    "needs_planning": False,
                    "exploratory": True,
                    "very_complex": False,
                }
            )

    def explain_recommendation(self, recommendation: PatternRecommendation) -> str:
        """
        Generate a human-readable explanation of the recommendation.

        Args:
            recommendation: The pattern recommendation

        Returns:
            Formatted explanation string
        """
        pattern_emoji = "🤖" if recommendation.pattern_type == "agent" else "🔄"

        # Get pattern description
        descriptions = {
            "react": "Iterative reasoning with tool use - adapts based on results",
            "lats": "Tree search with self-reflection - explores multiple solution paths",
            "prompt-chain": "Sequential LLM calls with validation gates",
            "routing": "Classifies input and routes to specialized handler",
            "rewoo": "Plans all steps upfront, executes in parallel",
            "plan-execute": "Adaptive planning with replanning capability",
        }

        desc = descriptions.get(recommendation.pattern_name, "Unknown pattern")

        # Format characteristics
        detected = [k for k, v in recommendation.characteristics.items() if v]
        chars_text = ", ".join(detected) if detected else "none"

        explanation = f"""
{pattern_emoji} **Recommended: {recommendation.pattern_name}** ({recommendation.pattern_type.upper()})
Confidence: {recommendation.confidence:.0%}

**Why:** {recommendation.reasoning}

**Pattern:** {desc}

**Detected characteristics:** {chars_text}
"""
        return explanation.strip()


def create_confirmation_prompt(recommendation: PatternRecommendation, selector: PatternSelector) -> str:
    """
    Create a user confirmation prompt.

    Args:
        recommendation: The pattern recommendation
        selector: The selector instance (for explanation)

    Returns:
        Formatted confirmation prompt
    """
    explanation = selector.explain_recommendation(recommendation)

    prompt = f"""
{explanation}

**Continue with {recommendation.pattern_name}?**
- Yes: Proceed with this pattern
- No: Use default ReAct agent instead
- Change: Manually select a different pattern
"""
    return prompt.strip()
