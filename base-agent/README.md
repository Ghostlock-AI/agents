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

- gpt4o-mini LLM
- Tavily Search Tool
- Shell Tool
- reAct planning pattern
