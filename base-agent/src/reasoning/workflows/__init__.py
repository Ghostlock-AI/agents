"""
Workflow Patterns

Workflows involve LLMs and tools orchestrated through predefined code paths.
They provide predictability and structure for well-defined tasks.

Available Workflows:
- ReWOO: Plans all steps upfront, executes in parallel (Orchestrator-Worker)
- Plan-Execute: Adaptive planning with replanning (Evaluator-Optimizer)

When to use workflows:
- Tasks with predictable, well-defined steps
- Research and data gathering from multiple sources
- Multi-step tasks where parallelization provides speedup
- Situations requiring clear progress tracking and transparency
"""

from reasoning.workflows.rewoo import ReWOOStrategy
from reasoning.workflows.plan_execute import PlanExecuteStrategy

__all__ = ["ReWOOStrategy", "PlanExecuteStrategy"]
