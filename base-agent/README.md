# Base Agent

A foundational AI agent built with LangGraph, featuring session memory and streaming responses.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your OpenAI API key:

```
OPENAI_API_KEY=your_actual_api_key_here
```

### 3. Run the Agent

```bash
python src/main.py
```

## Features

- **Session Memory**: Maintains conversation context across messages
- **Streaming Responses**: Real-time token-by-token output
- **Simple Interface**: Clean chat prompt with exit commands

## Usage

```
> Hello!
Assistant: Hi there! How can I help you today?

> What's 2+2?
Assistant: 2 + 2 equals 4.

> exit
```

Type `exit`, `quit`, or `q` to end the conversation.

## Architecture

This is Phase 1 of the base agent implementation:
- LangGraph for state management
- OpenAI LLM integration
- In-memory session persistence
- Simple chat loop

See `base_agent_design.md` for the full architecture plan.

## Next Steps

Future phases will add:
- Advanced reasoning (LATS/MCTS)
- Tool system (shell, search, files)
- Context management & file deduplication
- ChromaDB long-term memory
- Multi-agent orchestration
