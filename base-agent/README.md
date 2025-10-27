# Base Agent

A foundational AI agent built with LangGraph, featuring session memory and streaming responses.

## Quick Start

### Option 1: Docker

```bash
# 1. Create .env file
cp .env.example .env
# Edit .env and add your API keys

# 2. Start container (drops into shell)
docker compose up -d
docker exec -it bash

# 3. Inside container, install and run:
pip install -r requirements.txt
python3 src/main.py

# Stop when done
# Ctrl+C to exit, then:
docker compose down
```

### Option 2: Local Python

```bash
# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env
# Edit .env and add your API keys

# 4. Run the agent
python3 src/main.py
```

## Features

### LLM Support
- **OpenAI**: GPT-4, GPT-4o, GPT-4o-mini (default)
- More providers coming: Anthropic Claude, local models (vLLM, Ollama)

### Tools
- **Web Search**: DuckDuckGo search (no API key required)
- **Web Fetch**: Extract content from URLs
- **Shell Execution**: Run system commands (sandboxed)

### Reasoning: Workflows vs Agents 🧠

Following [Anthropic's best practices](https://www.anthropic.com/engineering/building-effective-agents), we distinguish between two fundamental patterns:

**🔄 WORKFLOWS** - Predefined code paths with LLM orchestration:
- **Prompt Chain**: Sequential LLM calls with validation gates
- **Routing**: Classifies inputs and routes to specialized handlers
- **ReWOO**: Plans all steps upfront, executes in parallel
- **Plan-Execute**: Adaptive planning with replanning capability

**🤖 AGENTS** - LLMs dynamically control their own processes:
- **ReAct** (Default): Iterative reasoning with tool use
- **LATS**: Tree search with self-reflection (advanced)

#### When to Use Which?

```
Is the task open-ended with unpredictable steps?
├─ YES → Use an AGENT (ReAct or LATS)
└─ NO  → Use a WORKFLOW
         ├─ Need quality gates? → Prompt Chain
         ├─ Multiple input types? → Routing
         ├─ Parallel execution? → ReWOO
         └─ Adaptive planning? → Plan-Execute
```

**Quick Guide:**
- 📝 Content generation → **Prompt Chain**
- 🎯 Customer service → **Routing**
- 🔬 Research tasks → **ReWOO**
- 🏗️ Complex projects → **Plan-Execute**
- 🔄 General queries → **ReAct** (default)
- 🌳 Hard problems → **LATS**

### Intelligent Pattern Selection
- **Automatic**: AI analyzes your query and recommends the best reasoning pattern
- **User Confirmation**: You review and approve (or override) the recommendation
- **Based on Anthropic's Framework**: Uses task characteristics to select optimal approach
- **Transparent**: Shows confidence score and reasoning for each recommendation

### Interactive TUI
- Multiline input with syntax highlighting
- File attachment support (`/file <path>`)
- Rich markdown rendering
- Runtime strategy switching
- Session memory across conversations

## Usage

### Interactive Mode

```bash
python src/main.py
```

**How it works:**
1. Type your query
2. The AI analyzes your query and recommends a reasoning pattern
3. You see the recommendation with confidence score and explanation
4. Choose:
   - `y` - Switch to recommended pattern and proceed
   - `n` - Keep current pattern and proceed
   - `c` - Cancel the query

Example:
```
▌ Help me debug my Python script that's crashing

🤖 Recommended: react (AGENT)
Confidence: 90%

Why: The task is exploratory and requires investigation...

Options:
  y - Switch to react and proceed
  n - Keep current and proceed
  c - Cancel this query
```

The TUI supports these commands:

**File Operations:**
```
/file <path>          Attach a file to your message
```

**Reasoning Strategy:**
```
/reasoning list       List all available strategies
/reasoning current    Show current strategy
/reasoning switch <name>   Switch to: react, rewoo, plan-execute, or lats
/reasoning info [name]     Show detailed info about a strategy
```

**Input Mode:**
```
/enter send           Send on Enter (default)
/enter newline        Newline on Enter, double-newline to send
```

**Exit:**
```
/quit                 Exit the application
```

### One-Shot Mode

Run a single query and exit:

```bash
python src/main.py "What is the weather in San Francisco?"
```

### Reasoning Strategy Examples

**For general queries** (default):
```
/reasoning switch react
```

**For research tasks** (parallel execution):
```
/reasoning switch rewoo
Tell me about the history of AI and current state of LLMs
```

**For complex multi-step tasks**:
```
/reasoning switch plan-execute
Create a complete Python project structure for a web API
```

**For complex problems requiring exploration**:
```
/reasoning switch lats
Optimize this sorting algorithm for performance
```

### Testing

Run the test suite to verify all strategies:

```bash
python test_strategies.py
```
