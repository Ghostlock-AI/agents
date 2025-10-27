# Workflow System Guide

## Overview

The base-agent now supports **two distinct modes**:

1. **Agent Generalist Mode** - Autonomous agents that reason iteratively
2. **Workflow Mode** - Sequential LLM pipelines with predefined steps

This design follows [Anthropic's recommendations](https://www.anthropic.com/engineering/building-effective-agents) for building effective AI systems.

---

## Agent vs Workflow: When to Use Each

### Use Agent Mode When:
- ✅ You need dynamic, adaptive reasoning
- ✅ The solution path is unclear upfront
- ✅ The agent should control its own tool usage
- ✅ Tasks require exploration and self-correction

**Example:** "Debug this codebase and fix any issues you find"

### Use Workflow Mode When:
- ✅ You have a clear, fixed sequence of tasks
- ✅ Each step has a specific, well-defined job
- ✅ You want predictability and transparency
- ✅ Tasks don't need dynamic adaptation

**Example:** "Research topic → Analyze findings → Draft summary"

---

## Agent Generalist Mode

Agent mode provides autonomous reasoning with self-directed tool use.

### Available Strategies

| Strategy | Description | Best For |
|----------|-------------|----------|
| **ReAct** | Iterative reasoning with tool use | General tasks, exploration |
| **ReWOO** | Plan upfront, execute in parallel | Research, data gathering |
| **Plan-Execute** | Adaptive planning with replanning | Complex multi-step tasks |
| **LATS** | Tree search with self-reflection | Complex problems requiring exploration |

### Usage

```python
from agent import Agent

# Create agent in generalist mode
agent = Agent(mode="agent", reasoning_strategy="react")

# Query the agent
for chunk in agent.stream("What's the weather in San Francisco?"):
    print(chunk, end='', flush=True)

# Switch strategies at runtime
agent.switch_reasoning_strategy("rewoo")
```

### Agent Mode Features

- **Dynamic Tool Selection** - Agent decides which tools to use
- **Iterative Reasoning** - Thinks, acts, observes, repeats
- **Self-Correction** - Can adapt based on results
- **Session Memory** - Maintains conversation context

---

## Workflow Mode

Workflow mode provides sequential LLM pipelines where each step performs exactly one job.

### Architecture

```
User Input → Step 1 LLM → Step 2 LLM → Step 3 LLM → Synthesizer → User Output
```

Each step:
- Gets input from previous step (or user query for first step)
- Performs its specific job (research, analyze, draft, etc.)
- Passes output to next step
- Final synthesizer creates polished user-facing response

### Building Custom Workflows

```python
from agent import Agent
from workflow_system import WorkflowBuilder

# Create agent in workflow mode
agent = Agent(mode="workflow")

# Build a custom workflow
workflow = (WorkflowBuilder("my_workflow")
            .add_step("research", "Gather relevant information")
            .add_step("analyze", "Identify key insights")
            .add_step("draft", "Write initial summary")
            .set_synthesizer("Polish and format for user")
            .build())

agent.set_workflow(workflow)

# Execute workflow
for chunk in agent.stream("Tell me about LangGraph"):
    print(chunk, end='', flush=True)
```

### Using Workflow Templates

Pre-built workflows for common patterns:

```python
from agent import Agent

agent = Agent(mode="workflow")

# Load a template
agent.load_workflow_template("research_and_summarize")

# Execute
for chunk in agent.stream("Explain quantum computing"):
    print(chunk, end='', flush=True)
```

### Available Templates

**1. research_and_summarize**
- Step 1: Research topic using tools
- Step 2: Analyze findings and extract insights
- Step 3: Draft clear summary
- Synthesizer: Polish for user

**2. code_review**
- Step 1: Understand code structure
- Step 2: Identify issues (bugs, style, security)
- Step 3: Suggest improvements
- Synthesizer: Format review report

### Workflow Features

- **Transparency** - See exactly which step is executing
- **Predictability** - Fixed sequence, no surprises
- **Debugging** - Inspect execution trace
- **Tool Support** - Steps can use tools when needed

### Execution Trace

```python
# After workflow execution, inspect the trace
trace = agent.get_workflow_trace()

for entry in trace:
    print(f"Step {entry['step']}: {entry['name']}")
    print(f"  Input: {entry.get('input', 'N/A')[:100]}")
    print(f"  Output: {entry.get('output', 'N/A')[:100]}")
```

---

## Complete Examples

### Example 1: Agent Mode (Dynamic)

```python
from agent import Agent

# Agent decides how to solve the problem
agent = Agent(mode="agent", reasoning_strategy="react")

query = "Find the latest Python release and tell me what's new"

for chunk in agent.stream(query):
    print(chunk, end='', flush=True)

# Agent will:
# 1. Decide to search the web
# 2. Find Python release page
# 3. Extract key features
# 4. Summarize findings
# All autonomously!
```

### Example 2: Workflow Mode (Sequential)

```python
from agent import Agent
from workflow_system import WorkflowBuilder

# You define exact steps
agent = Agent(mode="workflow")

workflow = (WorkflowBuilder("release_notes")
            .add_step("search", "Search for latest Python release")
            .add_step("extract", "Extract version number and key features")
            .add_step("summarize", "Write brief summary of changes")
            .set_synthesizer("Format as clear, concise release notes")
            .build())

agent.set_workflow(workflow)

query = "Find the latest Python release and tell me what's new"

for chunk in agent.stream(query):
    print(chunk, end='', flush=True)

# Workflow will:
# 1. Execute search step
# 2. Pass results to extract step
# 3. Pass extracted data to summarize step
# 4. Synthesize final output
# Exactly as defined!
```

---

## Advanced: Steps with Tools

Workflow steps can use tools when needed:

```python
from agent import Agent
from workflow_system import WorkflowBuilder

agent = Agent(mode="workflow")

# Step that uses shell_exec tool
workflow = (WorkflowBuilder("date_info")
            .add_step(
                "get_date",
                "Use the shell_exec tool to run 'date' command and get current date"
            )
            .add_step(
                "explain",
                "Explain what day of week it is and any interesting facts"
            )
            .set_synthesizer("Format in friendly, conversational way")
            .build())

agent.set_workflow(workflow)

for chunk in agent.stream("Tell me about today"):
    print(chunk, end='', flush=True)
```

The LLM in step 1 will call `shell_exec` to get real data, then pass it to step 2.

---

## Design Philosophy

This two-mode design aligns with Anthropic's guidance:

### From "Building Effective Agents"

> **Workflows** involve "LLMs and tools orchestrated through predefined code paths"
>
> **Agents** represent systems where "LLMs dynamically direct their own processes and tool usage"

### Key Principles

1. **Simplicity** - Workflows are simpler than agents
2. **Transparency** - Workflow steps are visible
3. **Composability** - Build complex systems from simple patterns
4. **Right Tool for the Job** - Choose based on task characteristics

---

## Migration Guide

### If You're Currently Using:

**ReAct/LATS** → These are now "agent mode" strategies
- No code changes needed
- Works exactly the same
- Can still switch strategies

**ReWOO/Plan-Execute** → These remain in agent mode
- They're still autonomous (agent controls execution)
- Consider using workflow mode for truly fixed sequences

### When to Migrate to Workflows:

✅ **Migrate if:**
- You have fixed, repetitive processes
- You want to see exactly what's happening at each step
- You're building pipelines (extract → transform → load style)

❌ **Stay with agents if:**
- You need adaptive reasoning
- Solution path is unclear
- Agent should explore and self-correct

---

## API Reference

### Agent Class

```python
Agent(
    mode: str = "agent",           # "agent" or "workflow"
    reasoning_strategy: str = "react",  # Only for agent mode
    model_name: str = None,        # OpenAI model
    temperature: float = 0.7,
    tools: list = None
)
```

### WorkflowBuilder

```python
WorkflowBuilder(name: str)
    .add_step(name: str, instruction: str)
    .set_synthesizer(instruction: str)
    .build() -> Workflow
```

### Agent Methods (Workflow Mode)

```python
agent.set_workflow(workflow: Workflow)
agent.load_workflow_template(template_name: str)
agent.get_workflow_trace() -> List[Dict]
```

---

## Troubleshooting

### "No workflow set" error

```python
# Make sure to set a workflow first
agent = Agent(mode="workflow")
agent.load_workflow_template("research_and_summarize")  # Required!
agent.stream("Your query")
```

### Steps not using tools

```python
# Make sure your instruction explicitly mentions the tool
.add_step(
    "get_data",
    "Use the shell_exec tool to run 'date' command"  # Explicit!
)
```

### Agent vs Workflow confusion

```python
# Agent mode
agent = Agent(mode="agent")  # No workflow needed
agent.stream("Query")  # Uses reasoning strategy

# Workflow mode
agent = Agent(mode="workflow")  # Must set workflow
agent.load_workflow_template("...")  # Required!
agent.stream("Query")  # Uses workflow steps
```

---

## What's Next?

### Planned Features

1. **Conditional Workflows** - If/else branching in workflows
2. **Parallel Steps** - Execute multiple steps simultaneously
3. **More Templates** - data_pipeline, content_generation, etc.
4. **Workflow Composition** - Combine workflows into larger systems
5. **Human-in-the-Loop** - Approval gates between steps

### Contributing

Want to add a workflow template? See `src/workflow_system.py` - `WorkflowTemplates` class.

---

## Summary

- **Agent Mode** = Autonomous reasoning (ReAct, ReWOO, Plan-Execute, LATS)
- **Workflow Mode** = Sequential pipelines (Step 1 → Step 2 → Step 3 → Synthesizer)
- **Choose Based on Task** = Fixed sequence? Workflow. Dynamic exploration? Agent.
- **Both Available** = Switch modes as needed for different tasks

This design gives you the best of both worlds: autonomous agents when you need adaptability, predictable workflows when you need transparency.
