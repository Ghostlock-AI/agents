import os

from dotenv import load_dotenv
from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    InferenceClientModel,
    VisitWebpageTool,
)

# Load environment variables from .env file
load_dotenv()


def create_agent():
    """Create and return a configured smolagent instance."""
    # Get HuggingFace API token from environment
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    if not hf_token:
        raise ValueError("HUGGINGFACE_API_TOKEN environment variable not set")

    # Initialize the model with a specific model ID
    # Using Qwen2.5-Coder which is good for code-based agents
    model = InferenceClientModel(
        model_id="Qwen/Qwen2.5-Coder-32B-Instruct", token=hf_token
    )

    # Create agent with some basic tools
    agent = CodeAgent(
        tools=[DuckDuckGoSearchTool(), VisitWebpageTool()],
        model=model,
    )

    return agent


def main():
    """Run a simple one-off task with the agent."""
    agent = create_agent()

    # Run a simple task
    result = agent.run("What is the weather like in San Francisco today?")
    print(result)


if __name__ == "__main__":
    main()
