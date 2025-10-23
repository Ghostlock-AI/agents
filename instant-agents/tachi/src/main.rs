use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use tera::Tera;

// CLI
#[derive(Parser)]
#[command(name = "tachi", version, about = "Compile YAML ->Python smolagent")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Generate a Python agent from YAML spec
    Gen {
        /// Path to YAML spec
        input: PathBuf,
        /// Input directory (defaults to current directory)
        #[arg(short, long, default_value = ".")]
        dir: PathBuf,
        /// Output directory (defaults to current directory)
        #[arg(short, long, default_value = ".")]
        out: PathBuf,
        /// Overwrite existing files if present
        #[arg(long)]
        force: bool,
    },
}

// SPEC
#[derive(Debug, Serialize, Deserialize)]
struct Spec {
    agent: Agent,
}

#[derive(Debug, Serialize, Deserialize)]
struct Agent {
    name: String,
    tools: Vec<Tool>,
    model: Model,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum Tool {
    #[serde(alias = "search")]
    Search,
    #[serde(alias = "webpage")]
    Webpage,
}

impl Tool {
    fn py_import_name(&self) -> &'static str {
        match self {
            Tool::Search => "DuckDuckGoSearchTool",
            Tool::Webpage => "VisitWebpageTool",
        }
    }
    fn py_instance(&self) -> &'static str {
        match self {
            Tool::Search => "DuckDuckGoSearchTool()",
            Tool::Webpage => "VisitWebpageTool()",
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
enum Model {
    #[serde(alias = "qwen-coder")]
    QwenCoder,
}

impl Model {
    fn model_id(&self) -> &'static str {
        match self {
            Model::QwenCoder => "Qwen/Qwen2.5-Coder-32B-Instruct",
        }
    }
}

// ----------------------
// Template (embedded)
// ----------------------

const PY_AGENT_TEMPLATE: &str = r#"import os
from dotenv import load_dotenv
from smolagents import InferenceClientModel, CodeAgent, {{ tool_imports | join(sep=", ") }}

# Load environment variables from .env file
load_dotenv()

def create_agent():
    """Create and return a configured smolagents instance."""
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not hf_token:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN environment variable not set")

    model = InferenceClientModel(
        model_id="{{ model_id }}",
        token=hf_token
    )

    agent = CodeAgent(
        tools=[{% for t in tool_instances %}{{ t }}{% if not loop.last %}, {% endif %}{% endfor %}],
        model=model,
    )
    return agent
"#;

const PY_CLI_TEMPLATE: &str = r#"#!/usr/bin/env python3
"""
Interactive CLI for the smolagent.
Provides a classic chat interface with input/output loop.
"""

import sys
from agent import create_agent


def print_banner():
    """Print welcome banner."""
    print("=" * 60)
    print("HuggingFace Smolagent CLI")
    print("=" * 60)
    print("Type your requests and press Enter.")
    print("Type 'exit', 'quit', or press Ctrl+C to exit.")
    print("=" * 60)
    print()


def main():
    """Run the interactive CLI loop."""
    try:
        # Initialize agent once at startup
        print("Initializing agent...")
        agent = create_agent()
        print("Agent ready!\n")

        print_banner()

        # Main interaction loop
        while True:
            try:
                # Get user input
                user_input = input("\nYou: ").strip()

                # Check for exit commands
                if user_input.lower() in ["exit", "quit", "q"]:
                    print("\nGoodbye!")
                    break

                # Skip empty inputs
                if not user_input:
                    continue

                # Run agent with user input
                print("\nAgent: ", end="", flush=True)
                result = agent.run(user_input)
                print(result)

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except EOFError:
                print("\n\nGoodbye!")
                break

    except Exception as e:
        print(f"\nError initializing agent: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
"#;

fn write_file(out_dir: &Path, name: &str, content: &str, force: bool) -> Result<PathBuf> {
    fs::create_dir_all(out_dir)
        .with_context(|| format!("creating output directory {}", out_dir.display()))?;
    let path = out_dir.join(name);
    if path.exists() && !force {
        anyhow::bail!(
            "refusing to overwrite existing file: {} (use --force)",
            path.display()
        );
    }
    fs::write(&path, content).with_context(|| format!("writing {}", path.display()))?;
    Ok(path)
}

fn render_agent_py(spec: &Spec) -> Result<String> {
    let mut ctx = tera::Context::new();
    let tool_imports: Vec<String> = spec
        .agent
        .tools
        .iter()
        .map(|t| t.py_import_name().to_string())
        .collect();
    let tool_instances: Vec<String> = spec
        .agent
        .tools
        .iter()
        .map(|t| t.py_instance().to_string())
        .collect();

    ctx.insert("tool_imports", &tool_imports);
    ctx.insert("tool_instances", &tool_instances);
    ctx.insert("model_id", &spec.agent.model.model_id());

    // render one-off template from the embedded string
    let py = Tera::one_off(PY_AGENT_TEMPLATE, &ctx, false).context("rendering agent.py template")?;
    Ok(py)
}

// MAIN
fn main() -> Result<()> {
    let cli = Cli::parse();

    let Commands::Gen { input, dir: _, out, force } = cli.command;
    let yaml =
        fs::read_to_string(&input).with_context(|| format!("reading {}", input.display()))?;
    let spec: Spec = serde_yaml_ng::from_str(&yaml).context("parsing YAML")?;

    // Create project directory with agent name
    let project_dir = out.join(&spec.agent.name);

    // Generate agent.py
    let agent_py = render_agent_py(&spec)?;
    write_file(&project_dir, "agent.py", &agent_py, force)?;

    // Generate cli.py
    write_file(&project_dir, "cli.py", PY_CLI_TEMPLATE, force)?;

    // Generate requirements.txt
    let reqs = "smolagents\npython-dotenv\nddgs\n";
    write_file(&project_dir, "requirements.txt", reqs, force)?;

    // Generate .env.example
    let env = "# Put your Hugging Face token here\nHUGGINGFACEHUB_API_TOKEN=\n";
    write_file(&project_dir, ".env.example", env, force)?;

    println!("✔ Generated {} project", spec.agent.name);

    Ok(())
}
