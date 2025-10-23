import os
from dotenv import load_dotenv
from smolagents import InferenceClientModel, CodeAgent, DuckDuckGoSearchTool, VisitWebpageTool

# Load environment variables from .env file
load_dotenv()

def create_agent():
    """Create and return a configured smolagents instance."""
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN environment variable not set")

    model = InferenceClientModel(
        model_id="Qwen/Qwen2.5-Coder-32B-Instruct",
        token=hf_token
    )

    agent = CodeAgent(
        tools=[DuckDuckGoSearchTool(), VisitWebpageTool()],
        model=model,
    )
    return agent
