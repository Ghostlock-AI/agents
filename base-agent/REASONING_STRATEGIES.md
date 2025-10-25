# Reasoning Strategies Guide

This document explains the reasoning strategies available in the base agent and when to use each one.

## Overview

The base agent supports multiple reasoning strategies that can be switched at runtime. Each strategy has different strengths and is optimized for different types of tasks.

## Available Strategies

### 1. ReAct (Reason + Act) [DEFAULT]

**Pattern:** Iterative reasoning with tool use

**How it works:**
1. Agent thinks about what to do
2. Agent acts (uses tools or responds)
3. Agent observes results
4. Repeat until done

**Best for:**
- General-purpose tasks
- Exploratory problem-solving
- Tasks where next steps depend on results
- Unknown complexity

**Performance:**
- Speed: Fast ⚡⚡⚡
- Quality: Good ✓✓✓
- Token usage: Low 💰
- Reliability: Very High ✓✓✓✓

**Configuration:**
- `max_iterations`: Maximum reasoning cycles (default: 20)

**Example use cases:**
- General Q&A
- Debug a specific error
- Research a topic with follow-up questions
- File operations and system tasks

---

### 2. ReWOO (Reasoning Without Observation)

**Pattern:** Plan everything upfront, execute in parallel

**How it works:**
1. Create complete plan of all steps
2. Execute all tool calls in parallel
3. Synthesize results into final answer

**Best for:**
- Research tasks with multiple queries
- Data gathering from multiple sources
- Predictable workflows
- Tasks where steps don't depend on each other

**Performance:**
- Speed: Very Fast ⚡⚡⚡⚡ (parallel execution)
- Quality: Good ✓✓✓
- Token usage: Low 💰
- Reliability: Medium ✓✓ (can't adapt to unexpected results)

**Limitations:**
- Cannot adapt plan based on results
- Requires clear problem structure
- Less effective for exploratory tasks

**Example use cases:**
- "Compare Python vs JavaScript for web development"
- "Gather information about top 5 AI companies"
- Multi-source research projects
- Batch operations

---

### 3. Plan-and-Execute

**Pattern:** Adaptive planning with sequential execution

**How it works:**
1. Create high-level plan
2. Execute steps sequentially
3. Check progress after each step
4. Replan if needed
5. Continue until goal achieved

**Best for:**
- Complex multi-step tasks
- Tasks requiring adaptive planning
- Projects needing decomposition
- Progress tracking matters

**Performance:**
- Speed: Medium ⚡⚡
- Quality: Very Good ✓✓✓✓
- Token usage: Medium 💰💰
- Reliability: High ✓✓✓

**Configuration:**
- `max_replans`: Maximum number of times to replan (default: 3)

**Limitations:**
- Slower than ReAct (more LLM calls)
- Can get stuck in planning loops
- Overkill for simple tasks

**Example use cases:**
- "Build a complete REST API with tests"
- "Refactor this codebase following SOLID principles"
- Project setup with multiple components
- Complex debugging requiring investigation

---

### 4. LATS (Language Agent Tree Search)

**Pattern:** Tree search with self-reflection

**How it works:**
1. Generate multiple candidate approaches
2. Reflect on and evaluate each candidate
3. Select best approach
4. Execute and repeat if needed

**Best for:**
- Complex algorithmic problems
- Code optimization tasks
- Problems with multiple valid approaches
- When quality matters more than speed

**Performance:**
- Speed: Slow ⚡ (many LLM calls)
- Quality: Excellent ✓✓✓✓✓
- Token usage: High 💰💰💰
- Reliability: Very High ✓✓✓✓

**Configuration:**
- `num_candidates`: Alternatives to consider at each step (default: 3)
- `max_depth`: Maximum search depth (default: 5)
- `enable_reflection`: Whether to use self-reflection (default: true)

**Limitations:**
- Much slower than other strategies
- Higher token/cost usage
- Overkill for simple tasks

**Example use cases:**
- "Optimize this sorting algorithm"
- "Find the best architecture for this system"
- Complex coding challenges
- HumanEval-style problems

---

## Choosing the Right Strategy

### Decision Tree

```
Is it a simple, general task?
├─ YES → Use ReAct (default)
└─ NO → Continue

Does it involve multiple independent searches/queries?
├─ YES → Use ReWOO (parallel speedup)
└─ NO → Continue

Is it a complex multi-step task requiring planning?
├─ YES → Use Plan-and-Execute
└─ NO → Continue

Do you need the absolute best solution?
└─ YES → Use LATS (accept slower speed)
```

### Quick Reference Table

| Task Type | Recommended Strategy | Alternative |
|-----------|---------------------|-------------|
| General Q&A | ReAct | - |
| Web Research | ReWOO | ReAct |
| Multi-source Data Gathering | ReWOO | ReAct |
| Complex Project Setup | Plan-and-Execute | ReAct |
| Code Refactoring | Plan-and-Execute | LATS |
| Algorithm Optimization | LATS | Plan-and-Execute |
| Debugging | ReAct | Plan-and-Execute |
| File Operations | ReAct | - |
| System Administration | ReAct | Plan-and-Execute |
| Exploratory Tasks | ReAct | LATS |

---

## Usage in TUI

### List Available Strategies
```
/reasoning list
```

### Check Current Strategy
```
/reasoning current
```

### Switch Strategy
```
/reasoning switch react
/reasoning switch rewoo
/reasoning switch plan-execute
/reasoning switch lats
```

### Get Strategy Details
```
/reasoning info react
/reasoning info lats
```

---

## Strategy Indicators

When the agent is thinking, you'll see the active strategy in the status message:

```
[REACT] Thinking...
[REWOO] Thinking...
[PLAN-EXECUTE] Thinking...
[LATS] Thinking...
```

---

## Extending with Custom Strategies

To create a custom reasoning strategy:

1. Create a new file in `src/reasoning/strategies/`
2. Inherit from `ReasoningStrategy` base class
3. Implement required methods:
   - `create_graph()`: Build your LangGraph
   - `get_name()`: Return strategy identifier
   - `get_description()`: Return human-readable description
4. Register in `strategy_registry.py`

Example:
```python
from .base import ReasoningStrategy

class MyCustomStrategy(ReasoningStrategy):
    def get_name(self) -> str:
        return "custom"

    def get_description(self) -> str:
        return "My custom reasoning approach"

    def create_graph(self, agent_state_class, llm_with_tools, tools):
        # Build your custom LangGraph here
        pass
```

---

## Performance Benchmarks

*Note: These are approximate and depend on task complexity and LLM model used.*

| Strategy | Avg Steps | Avg Tokens | Avg Time | Success Rate |
|----------|-----------|------------|----------|--------------|
| ReAct | 3-5 | 2K-5K | Fast | 85% |
| ReWOO | 2-3 | 2K-4K | Very Fast | 75% |
| Plan-and-Execute | 5-8 | 5K-10K | Medium | 90% |
| LATS | 8-15 | 10K-20K | Slow | 95% |

---

## Research References

- **ReAct**: [Yao et al., 2022](https://arxiv.org/abs/2210.03629)
- **ReWOO**: [Xu et al., 2023](https://arxiv.org/abs/2305.18323)
- **LATS**: [Zhou et al., 2023](https://arxiv.org/abs/2310.04406)

---

## Troubleshooting

**Strategy switch not working?**
- Check that you're using a valid strategy name (lowercase, hyphenated)
- Try `/reasoning list` to see available strategies

**Agent not behaving as expected?**
- Try switching to ReAct (most reliable)
- Check `/reasoning current` to confirm active strategy
- Use `/reasoning info` to see strategy configuration

**Too slow with LATS?**
- This is expected - LATS explores multiple paths
- Consider switching to Plan-and-Execute for similar quality, better speed
- Or use ReAct for simple tasks

**ReWOO failing on complex tasks?**
- ReWOO can't adapt its plan mid-execution
- Try Plan-and-Execute instead for adaptive planning
