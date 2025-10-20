I want to build an agent that is the baseline for a few others.
The agent will have these qualities

# Core components every agent needs:

1. LLM Interface (Claude Opus 4 or GPT-4)
2. Shell/System Executor (sandboxed)
3. Internet Search
4. Working Memory (current context)
5. Long-term Memory (start with Chroma, scale to Qdrant)
6. File Context Manager (version control)
7. Tool Registry (dynamic loading)
8. Planning Engine (ReAct → Pre-Act → LATS)

We will take this base and use it later to build
These three agents
Long-Running Agent: Use Plan-and-Execute pattern, MongoDB for persistent memory,
checkpoint/resume capabilities
Tool-Creating Agent: WebAssembly sandboxing (Wassette), test-driven validation,
pattern extraction from successes
Red-Teaming Agent: Attack pattern library, fuzzing capabilities, multi-turn
tracking, success metrics

What is essential in our base agent is:

- it can easily use a remote or locally running LLM, so we have to figure out how to
  to get it to work with local LLM's and how to interface with them.
- it needs a coherent context management system where files provided to the agent
  are

# Research notes on agent frameworks and features

Critical Discoveries

1. You're Right About LangChain
   LangChain has 96k GitHub stars but creates abstraction overhead that often obscures debugging. DSPy is powerful for experiment-heavy workflows with eval-driven iteration, achieving 40-50% performance improvements through automatic prompt optimization. However, for production, direct API calls provide the clarity and control you need. Best AI Agent Frameworks in 2025: Comparing LangGraph, DSPy, CrewAI, Agno, and More +2
2. What OpenAI/Anthropic Actually Use
   Anthropic uses a multi-agent orchestrator-worker pattern where a lead agent (Opus 4) coordinates specialized subagents (Sonnet 4) in parallel. They found that token usage alone explains 80% of performance variance - multi-agent systems use 15x more tokens than single agents but deliver 90% better results. How we built our multi-agent research system \ Anthropic +2
   OpenAI's Agents SDK uses a Python-first design with minimal abstractions - normal Python functions rather than complex DSLs. They're deprecating the Assistants API in 2026 in favor of the simpler Responses API with built-in Web Search, File Search, and Computer Use tools. Prompt HubMedium
3. Context Engineering Is Everything
   The KV-cache hit rate is the single most important metric for production agents. Anthropic emphasizes "progressive disclosure" - agents discover context through exploration rather than front-loading. Manus found that keeping prompt prefixes stable and using append-only context strategies are critical for performance. AnthropicManus
   LangChain identifies four key strategies: Write (save intermediate results outside context), Select (smart retrieval), Compress (summarization at boundaries), and Isolate (multi-agent with separate contexts). The distinction between memory (persistent storage) and context (working memory) is crucial. Context Engineering +2
4. File Context Management
   Your insight about keeping only the newest file versions is spot-on. Manus uses controlled variation in actions to prevent pattern fixation, but maintains strict file version control. They treat the filesystem as infinite memory, loading only summaries into context while keeping full content accessible via paths. Context Engineering for AI Agents: Lessons from Building Manus
5. Planning Methods Evolution
   Pre-Act (multi-step planning with reasoning) outperforms basic ReAct by generating detailed plans upfront. LATS (Language Agent Tree Search) using MCTS achieves 15-20% better performance than ReAct on complex tasks. Plan-and-Execute allows parallel execution and only requires the large LLM for planning steps. Pre-Act: Multi-Step Planning and Reasoning Improves Acting in LLM Agents +2
6. Self-Coding Agents
   L2MAC (ICLR 2024) demonstrates agents that detect invalid code and automatically error-correct while persisting a complete file-store memory. Claude Code can analyze evaluation results and refactor multiple tools simultaneously. The key pattern is: generate spec → implement → test → sandbox execute → self-correct → register tool. GitHub - e2b-dev/awesome-ai-agents: A list of AI autonomous agents +2
7. Memory Systems
   Modern agents use three-tier memory: Working Memory (8-32k tokens for immediate context), Short-term Memory (session-scoped via checkpointers), and Long-term Memory (persistent via vector DBs). The RAISE architecture mirrors human memory with scratchpad (short-term) and retrieval (long-term) components. Evaluating Very Long-Term Conversational Memory of LLM Agents | Request PDF +2
   Practical Recommendations
   Start Simple
   Based on the research, start with direct API calls rather than heavy frameworks. Both OpenAI and Anthropic use minimal abstractions internally. Add DSPy later for prompt optimization (40-50% improvements) once you have working baselines.
   Context is King

Keep prompt prefixes stable for KV-cache optimization
Use append-only context strategies
Implement smart file versioning (one version at a time)
Progressive loading beats front-loading everything

# Three agents we will build on top of base

Long-Running Agent: Use Plan-and-Execute pattern, MongoDB for persistent memory, checkpoint/resume capabilities
Tool-Creating Agent: WebAssembly sandboxing (Wassette), test-driven validation, pattern extraction from successes
Red-Teaming Agent: Attack pattern library, fuzzing capabilities, multi-turn tracking, success metrics
The research confirms your intuition: agents are conceptually simple but require meticulous context engineering. The difference between good and great agents isn't the framework - it's how well you manage context, enable learning, and optimize token usage.
