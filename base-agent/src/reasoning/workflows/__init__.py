"""
Workflow Patterns

Workflows involve LLMs and tools orchestrated through predefined code paths.
They provide predictability and structure for well-defined tasks.

Available Workflows:
- Prompt Chain: Sequential LLM calls with validation gates
- Routing: Classifies inputs and routes to specialized handlers
- ReWOO: Plans all steps upfront, executes in parallel (Orchestrator-Worker)
- Plan-Execute: Adaptive planning with replanning (Evaluator-Optimizer)

When to use workflows:
- Tasks with predictable, well-defined steps
- Content pipelines requiring quality gates
- Diverse inputs needing specialized handling
- Research and data gathering from multiple sources
- Multi-step tasks where parallelization provides speedup
- Situations requiring clear progress tracking and transparency
"""

from reasoning.workflows.prompt_chain import PromptChainWorkflow
from reasoning.workflows.routing import RoutingWorkflow
from reasoning.workflows.rewoo import ReWOOStrategy
from reasoning.workflows.plan_execute import PlanExecuteStrategy

__all__ = [
    "PromptChainWorkflow",
    "RoutingWorkflow",
    "ReWOOStrategy",
    "PlanExecuteStrategy",
]
