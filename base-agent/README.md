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

### Reasoning Strategies 🧠
Switch between multiple reasoning strategies at runtime:

- **ReAct** (Default): Iterative reasoning with tool use. Best for general tasks and exploration.
- **ReWOO**: Plans all steps upfront, executes in parallel. Best for research and data gathering.
- **Plan-and-Execute**: Creates adaptive plans with replanning. Best for complex multi-step tasks.
- **LATS**: Tree search with self-reflection. Best for complex problems requiring exploration (slower, higher quality).

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
