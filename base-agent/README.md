# Base Agent

A foundational AI agent built with LangGraph, featuring session memory and streaming responses.

## Quick Start

### 1. Install Dependencies

```bash
touch .env
{OPENAI_API_KEY=xxx}
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 src/main.py
```

## Features

- chat interface - DONE
- context management
- tool registry
- session persistence
- single agent ReAct loop
