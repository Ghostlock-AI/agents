# Simple HuggingFace Smolagent

A basic implementation of a HuggingFace smolagent with web search capabilities.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your HuggingFace API token:
```bash
export HUGGINGFACE_API_TOKEN=your_token_here
```

You can get a HuggingFace token from: https://huggingface.co/settings/tokens

## Required API Keys

- **HUGGINGFACE_API_TOKEN**: Required for accessing HuggingFace models through their API

## Usage

### Interactive CLI Mode (Recommended)

Run the interactive CLI for a classic chat interface:
```bash
python cli.py
```

This will start an interactive session where you can:
- Type questions and get responses
- Have multiple back-and-forth interactions
- Type `exit`, `quit`, or press Ctrl+C to exit

Example session:
```
You: What is the weather like in San Francisco today?
Agent: [searches and responds]

You: Tell me about the Golden Gate Bridge
Agent: [searches and responds]

You: exit
```

### One-off Mode

Run a single predefined task:
```bash
python simple_agent.py
```

You can modify the task in `simple_agent.py:30` to experiment with different queries.

## Features

The agent comes with:
- DuckDuckGo search tool
- Web page visiting tool
- Interactive CLI interface
- Reusable agent creation function
