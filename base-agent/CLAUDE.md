# Base Agent

A foundational AI agent framework built with LangGraph, designed to serve as the basis for specialized agent systems with advanced reasoning, tool use, and memory capabilities.

## Current State (Phase 1 - In Progress)

### ✅ Implemented Features

**Core Infrastructure:**
- **LangGraph ReAct Loop**: Reasoning + Action pattern for tool-based problem solving
- **Multi-LLM Support**: OpenAI integration (Anthropic & local models coming)
- **Tool System**: Web search (Tavily) and shell execution with safety guardrails
- **Session Memory**: Conversation context maintained via LangGraph's MemorySaver
- **CLI Interface**: Simple interactive chat with streaming responses
- **Tool Visibility**: `[TOOL: name]` notifications show when agent uses tools

**Architecture:**
- Modular design: `agent.py`, `reasoning.py`, `tools.py`, `cli.py`
- External system prompt loaded from file
- Environment-based configuration (.env)
- Type-safe state management with Pydantic

### 🚧 Current Limitations

- No file context management yet
- No long-term memory (ChromaDB not integrated)
- Single reasoning strategy (ReAct only)
- Limited to OpenAI models
- No session persistence to disk
- No sub-agent support

---

## Planned Features

### Phase 1: Core Foundation (Current Focus)

**System-Wide Installation**
- Package as installable CLI tool
- Global configuration management
- Multiple agent profiles

**LLM Provider Flexibility**
- `/model` command to swap models mid-conversation
- Support matrix:
  - OpenAI (GPT-4, GPT-4o, o1)
  - Anthropic (Claude Opus 4, Sonnet 4.5)
  - HuggingFace models
  - Local models (vLLM, Ollama)
- Easy local model configuration with endpoint specification

**Context Management System**
- File reference addition/removal via CLI
- Deduplication: Only current file versions in context
- View active files: `list_context()` command
- Smart file tracking with content hashing

**Advanced Tool Registry**
- Tool creation interface (write tools in natural language)
- Tool registration & validation
- Dynamic tool loading
- Tool usage analytics

**Session Persistence (SQLite)**
- Auto-save conversation state every N turns
- Resume sessions with full context restoration
- Session history browser
- Export/import conversations

**Reasoning Loop Selector**
- Swappable reasoning strategies:
  - ReAct (current)
  - LATS (Monte Carlo Tree Search)
  - Pre-Act (multi-step planning)
  - Plan-and-Execute
- `/reasoning <type>` to switch mid-session

**Prompt Engineering Repository**
- Curated collection of proven prompts
- Domain-specific prompt templates
- Meta-prompting: agents that seed other agents' prompts
- Prompt versioning & A/B testing

---

### Phase 2: Advanced Reasoning

**LATS Implementation**
- Monte Carlo Tree Search for complex problem solving
- Self-reflection at each decision node
- Exploration vs exploitation balancing
- 94.4% target on HumanEval benchmark

**Extended Thinking Protocol**
- Anthropic's "think before act" pattern
- Reasoning traces visible to user
- Token budget control for thinking depth

**Self-Reflection Mechanism**
- Automatic error detection & correction
- Solution quality evaluation
- Alternative approach consideration

**Sub-Agent Spawning**
- Manual sub-agent invocation
- Isolated context windows
- Structured output contracts
- Result aggregation

---

### Phase 3: Memory & Learning

**ChromaDB Integration**
- Persistent vector storage for task patterns
- Semantic search over past solutions
- Success/failure case library
- Code snippet retrieval

**Task Pattern Extraction**
- Automatic learning from successful executions
- Pattern recognition across tasks
- Complexity classification (simple/medium/complex)
- Tool usage pattern analysis

**Enrichment Retrieval System**
- Top-K similar task retrieval
- Solution scaffolds (step-by-step approaches)
- Error correction patterns
- Domain-specific enrichments

**Task Memory Tree (TMT)**
- Hierarchical task tracking
- State per node: pending/in_progress/completed/failed
- Learning extraction from execution paths
- Solves "dementia problem" - agent learns from experience

---

### Phase 4: Multi-Agent Orchestration

**Orchestrator-Worker Pattern**
- Lead agent (Opus 4 / GPT-4) for planning
- Worker agents (Sonnet 4.5 / GPT-4o-mini) for execution
- 15x token usage, 90% better results (Anthropic's findings)
- 90% time reduction on complex queries

**Parallel Sub-Agent Execution**
- Spawn 5-10 workers simultaneously
- Each with isolated context
- Specialized tool subsets per worker
- Concurrent task execution

**Result Aggregation**
- Structured output collection
- Conflict resolution
- Synthesis into coherent response
- Quality scoring

**Automatic Sub-Agent Selection**
- Task-to-agent matching
- Capability-based routing
- Load balancing
- Fallback strategies

---

### Phase 5: Optimization & Production

**KV-Cache Optimization**
- Stable prompt prefixes (80% performance gain)
- Append-only context strategy
- Progressive disclosure vs front-loading
- Cache hit rate monitoring (target: >70%)

**Context Compression**
- Token-efficient tool descriptions (dynamic loading)
- File summarization for unchanged content
- Smart retrieval (load only relevant sections)
- Emergency context offloading to sub-agents

**Specialized Sub-Agents**
- `code-reviewer`: Static analysis, pattern matching
- `test-generator`: Test creation & execution
- `security-analyzer`: Vulnerability scanning
- `documentation-writer`: Doc generation & validation
- `refactoring-specialist`: AST-based transformations

**Cost Tracking & Controls**
- Token usage per task
- Model selection optimization (10x cost reduction via workers)
- Budget limits & alerts
- Cache savings metrics (target: 50% reduction)

---

## Architecture Summary

```
base-agent/
├── src/
│   ├── main.py              # Entry point
│   ├── agent.py             # Agent class with ReAct loop
│   ├── reasoning.py         # LangGraph reasoning graphs
│   ├── tools.py             # Tool definitions & registry
│   ├── cli.py               # Command-line interface
│   └── system_prompt.txt    # Base system prompt
├── requirements.txt
├── .env.example
└── base_agent_design.md     # Comprehensive design doc
```

**Technology Stack:**
- **LangGraph**: State management & reasoning graphs
- **LangChain**: Tool integration (minimal usage)
- **OpenAI/Anthropic SDKs**: Direct API calls
- **ChromaDB → Qdrant**: Vector storage migration path
- **SQLite**: Session persistence
- **Pydantic**: Type safety & validation

---

## Design Principles

1. **Minimal Abstractions**: Direct API calls over heavy frameworks
2. **Context is King**: 80% of performance from context management
3. **Token Efficiency**: Optimize for KV-cache hits, compression strategies
4. **Progressive Disclosure**: Load context as needed, not upfront
5. **Learning-Enabled**: Extract patterns, improve over time
6. **Production-Ready**: Cost tracking, error handling, safety guardrails

---

## Success Metrics

**Performance:**
- Code generation: >90% on HumanEval
- Context efficiency: <20% token waste
- Multi-agent speedup: 5-10x on complex tasks

**Quality:**
- Error recovery: Self-correct within 3 attempts
- Test coverage: >80% for generated tests
- Memory retrieval: >85% precision (top-3 patterns)

**Cost:**
- Token budget: Tracked per-task
- Model selection: 10x savings via worker agents
- Cache optimization: 50% reduction target

---

## Future Extensions

Three specialized agents will be built on this base:

1. **Long-Running Agent**: Multi-day tasks, MongoDB persistence, progress APIs
2. **Tool-Creating Agent**: WebAssembly sandboxing, test-driven tool generation
3. **Red-Teaming Agent**: Attack patterns (MITRE ATT&CK), fuzzing, security metrics

---

## Contributing

This is a research project exploring state-of-the-art agent architectures based on:
- Anthropic's multi-agent research system
- LATS reasoning (ICML 2024)
- Task Memory Engine patterns
- Beads memory system (2025)

See `base_agent_design.md` for comprehensive design rationale and research findings.
