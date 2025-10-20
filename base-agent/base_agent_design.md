# BASE AGENT DESIGN: COMPREHENSIVE ARCHITECTURE PLAN

Based on extensive research into Anthropic's Claude Code, OpenAI's Agents SDK, and state-of-the-art agent frameworks, here's the complete design plan:

---

## 0. PROJECT STRUCTURE

```
base-agent/
├── .env                           # Environment variables (OPENAI_API_KEY, etc.)
├── .env.example                   # Template for environment setup
├── requirements.txt               # Python dependencies
├── README.md                      # Quick start guide
├── base_agent_design.md          # This file - comprehensive design doc
│
├── src/
│   ├── main.py                   # CLI entry point with chat loop
│   │
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── base.py               # LangGraph agent definition
│   │   ├── prompts.py            # System prompts & templates
│   │   └── orchestrator.py       # Sub-agent spawning logic
│   │
│   ├── reasoning/
│   │   ├── __init__.py
│   │   ├── graph.py              # LangGraph reasoning graph
│   │   └── lats.py               # LATS/MCTS implementation (Phase 2)
│   │
│   ├── context/
│   │   ├── __init__.py
│   │   └── manager.py            # File deduplication system
│   │
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── session.py            # Session state management
│   │   └── store.py              # ChromaDB integration (Phase 3)
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── registry.py           # Tool registry
│   │   ├── shell.py              # Shell executor
│   │   ├── search.py             # Web search
│   │   └── files.py              # File operations
│   │
│   └── models/
│       ├── __init__.py
│       └── schemas.py            # Pydantic models for state/tools
│
└── data/
    ├── sessions/                  # SQLite session databases
    └── chroma/                    # ChromaDB persistent storage
```

---

## 1. FRAMEWORK SELECTION

**Recommendation: Lightweight Custom Framework with LangGraph for Reasoning Graph**

**Rationale:**
- Anthropic and OpenAI both use minimal abstractions internally - direct API calls with Python-first design
- LangGraph specifically for the reasoning graph component (LATS/MCTS implementation)
- Avoid heavy LangChain abstractions - they create debugging overhead
- DSPy can be added later for prompt optimization (40-50% improvements) once baselines are established

**Core Stack:**
- **LLM Interface**: Direct API calls to Claude/OpenAI with fallback to local models (vLLM/Ollama)
- **Reasoning Engine**: LangGraph for LATS implementation (94.4% on HumanEval with GPT-4)
- **Memory**: ChromaDB → Qdrant migration path
- **Persistence**: SQLite for session state, file-based for agent definitions

---

## 2. BASE PROMPT ARCHITECTURE

### System Prompt Structure (Stable Prefix for KV-Cache Optimization)

```
ROLE & IDENTITY
├─ Agent type and specialization
├─ Core capabilities
└─ Operational constraints

REASONING FRAMEWORK
├─ Extended thinking protocol (think before acting)
├─ Tool usage patterns
├─ Error correction methodology
└─ Self-reflection triggers

CONTEXT MANAGEMENT RULES
├─ File reference handling
├─ Memory access patterns
├─ Progressive disclosure strategy
└─ Token budget awareness

MEMORY SCAFFOLDING
├─ Task pattern recognition
├─ Success pattern extraction
└─ Enrichment layer access
```

**Key Principles from Research:**
- **Stable prefixes**: Keep system prompt structure unchanged to maximize KV-cache hit rate (80% of performance variance)
- **Progressive disclosure**: Agent discovers context through exploration rather than front-loading
- **Append-only context**: Add new information without modifying existing context
- **Extended thinking**: Allow agent to write reasoning before acting (Anthropic's Lead Researcher pattern)

### Dynamic Prompt Components (Append-Only)

```
WORKING MEMORY (Current Session)
├─ Active file references (current version only)
├─ Recent tool outputs
└─ Intermediate results

RETRIEVED ENRICHMENTS (From ChromaDB)
├─ Similar task patterns
├─ Success strategies
└─ Error correction patterns

TASK STATE
├─ Current objective
├─ Subtask decomposition
└─ Progress tracking
```

---

## 3. REASONING GRAPH: LATS + ORCHESTRATOR-WORKER

### Core Architecture: Hybrid Approach

**Level 1: LATS for Single-Agent Complex Tasks**
- Monte Carlo Tree Search with LM-powered value functions
- Self-reflection at each node for exploration/exploitation balance
- Best for: Code generation, debugging, complex algorithmic tasks

**Level 2: Orchestrator-Worker for Multi-Agent Parallelization**
- Lead agent (stronger model: Opus 4 / GPT-4) for planning
- Worker agents (faster model: Sonnet 4 / GPT-4o-mini) for execution
- Best for: Multi-file refactoring, comprehensive testing, research tasks

### LATS Implementation Details

```
Graph Structure:
┌─────────────────┐
│  Generate Plan  │ (Extended Thinking)
└────────┬────────┘
         │
    ┌────▼────┐
    │  MCTS   │
    │  Loop   │
    └────┬────┘
         │
    ┌────▼─────────────────┐
    │ Selection (UCB)      │ ← Pick highest UCB node
    ├──────────────────────┤
    │ Expansion            │ ← Generate 5 candidate actions
    ├──────────────────────┤
    │ Simulation (Parallel)│ ← Execute all 5 in parallel
    ├──────────────────────┤
    │ Reflection           │ ← LM evaluates outcomes
    ├──────────────────────┤
    │ Backpropagation      │ ← Update node values
    └──────────────────────┘
         │
    ┌────▼────┐
    │ Best    │
    │ Action  │
    └─────────┘
```

**Key Parameters:**
- `nsim`: 5 simulations per iteration (balance speed/quality)
- `max_depth`: 10 for coding tasks
- Reward function: Code execution success + test coverage + self-reflection score

### Orchestrator-Worker Implementation

```
Lead Agent (Orchestrator)
├─ Query Analysis
├─ Strategy Development (Extended Thinking)
├─ Subtask Decomposition
├─ Worker Spawning (Parallel)
│   ├─ Worker 1: Objective, Output Format, Tools, Boundaries
│   ├─ Worker 2: Objective, Output Format, Tools, Boundaries
│   └─ Worker N: Objective, Output Format, Tools, Boundaries
└─ Result Aggregation & Synthesis

Each Worker:
├─ Isolated Context Window
├─ Interleaved Thinking (plan → act → reflect → refine)
├─ Tool Use (parallel when possible)
└─ Structured Output Return
```

**Performance Characteristics from Anthropic:**
- 15x more tokens than single agent
- 90% better results on complex tasks
- 90% reduction in research time for complex queries
- Token usage explains 80% of performance variance

**When to Use Each:**
- **LATS**: Single complex task requiring exploration (e.g., "optimize this algorithm")
- **Orchestrator-Worker**: Multiple parallel subtasks (e.g., "refactor entire module with tests")
- **Hybrid**: Lead uses LATS for planning, then spawns workers for execution

---

## 4. CONTEXT MANAGEMENT SYSTEM (THE CRITICAL COMPONENT)

### File Reference Deduplication Strategy

**Problem:** Multiple versions of same file pollute context window
**Solution:** Content-addressable file tracking with version control

```python
FileContextManager:
  ├─ file_registry: Dict[filepath, FileState]
  │   ├─ hash: str (content hash)
  │   ├─ version: int
  │   ├─ last_updated: timestamp
  │   ├─ reference_count: int
  │   └─ context_position: int (for stable ordering)
  │
  ├─ context_buffer: List[FileReference]
  │   └─ Ordered by: stability (system files first) → recency
  │
  └─ Methods:
      ├─ add_file(path, content) → Replaces old version if exists
      ├─ get_current_context() → Returns deduplicated file list
      ├─ summarize_unchanged() → Compress stable files to summaries
      └─ track_token_usage() → Monitor context window pressure
```

**Key Strategies from Research:**

1. **Write Strategy** (Manus approach)
   - Save intermediate results to filesystem
   - Load only summaries into context
   - Full content accessible via file paths

2. **Select Strategy** (Smart Retrieval)
   - Retrieve only relevant file sections
   - Use ChromaDB embeddings for semantic search
   - Priority: recently modified > high reference count > semantic similarity

3. **Compress Strategy**
   - Summarize files that haven't changed in N turns
   - Keep full content on disk, summary in context
   - Expand on-demand when needed

4. **Isolate Strategy** (Multi-Agent)
   - Each sub-agent has separate context
   - Lead agent only sees summaries from workers
   - No context pollution between workers

### Context Window Structure

```
┌─────────────────────────────────────┐
│ STABLE PREFIX (Never Changes)      │ ← Max KV-cache hits
│ - System prompt                    │
│ - Tool schemas                     │
│ - Agent identity                   │
├─────────────────────────────────────┤
│ FILE CONTEXT (Deduplicated)        │
│ - Current versions only            │
│ - Ordered by stability             │
│ - Summaries for unchanged files    │
├─────────────────────────────────────┤
│ MEMORY BLOCKS (Retrieved)          │
│ - Task enrichments                 │
│ - Success patterns                 │
│ - Similar solutions                │
├─────────────────────────────────────┤
│ WORKING MEMORY (Append-Only)       │
│ - Recent tool outputs              │
│ - Intermediate thoughts            │
│ - Current task state               │
└─────────────────────────────────────┘
```

**Token Budget Management:**
- Monitor: 70% threshold triggers compression
- Action: Summarize least-recently-used files
- Preserve: Tool outputs from last 3 turns
- Emergency: Offload to sub-agent with fresh context

---

## 5. LONG-TERM MEMORY & TASK ENRICHMENT (ChromaDB Architecture)

### Three-Tier Memory System

**Tier 1: Working Memory (In-Context)**
- 8-32k tokens
- Current task state
- Recent tool outputs
- Active file references

**Tier 2: Session Memory (SQLite)**
- Checkpoints for resume
- Task execution history
- Tool call logs
- File modification timeline

**Tier 3: Long-Term Memory (ChromaDB → Qdrant)**
- Task pattern embeddings
- Success/failure cases
- Code snippet library
- Enrichment metadata

### Task Enrichment & Scaffolding System

**Inspired by: Beads (2025) + Task Memory Engine**

```python
ChromaDB Collections:

1. task_patterns
   ├─ embedding: Task description vector
   ├─ metadata:
   │   ├─ task_type: "debugging" | "feature" | "refactor" | "test"
   │   ├─ success_rate: float
   │   ├─ avg_steps: int
   │   ├─ tools_used: List[str]
   │   └─ complexity: "simple" | "medium" | "complex"
   └─ content: Detailed pattern description

2. solution_scaffolds
   ├─ embedding: Problem vector
   ├─ metadata:
   │   ├─ language: str
   │   ├─ framework: str
   │   ├─ pattern_type: "architecture" | "algorithm" | "fix"
   │   └─ usage_count: int
   └─ content: Step-by-step approach + code examples

3. error_corrections
   ├─ embedding: Error message vector
   ├─ metadata:
   │   ├─ error_type: str
   │   ├─ root_cause: str
   │   └─ fix_success_rate: float
   └─ content: Diagnostic approach + fixes

4. code_snippets
   ├─ embedding: Functionality vector
   ├─ metadata:
   │   ├─ language: str
   │   ├─ tested: bool
   │   └─ dependencies: List[str]
   └─ content: Reusable code with tests
```

**Enrichment Flow:**

```
New Task Received
    │
    ├─ Embed task description
    │
    ├─ Query ChromaDB for:
    │   ├─ Similar task patterns (top 3)
    │   ├─ Relevant solution scaffolds (top 2)
    │   └─ Common error patterns (top 2)
    │
    ├─ Inject into context as "Memory Blocks"
    │
    ├─ Execute task with LATS/Orchestrator
    │
    └─ On completion:
        ├─ Extract success patterns
        ├─ Update usage counts
        └─ Store new patterns if novel
```

**Learning Loop (Task Memory Engine Pattern):**

```
Task Memory Tree (TMT):
├─ Root: High-level goal
├─ Children: Subtasks (hierarchical)
└─ Metadata per node:
    ├─ State: pending | in_progress | completed | failed
    ├─ Reasoning: Why this subtask
    ├─ Outcome: Success/failure with details
    └─ Learnings: What worked/didn't work

After task completion:
├─ Traverse TMT
├─ Extract successful paths
├─ Identify failure patterns
└─ Store enrichments in ChromaDB
```

**This solves:** "dementia/amnesia problem" - agent learns from past experiences

---

## 6. SUB-AGENT SYSTEM (Anthropic Pattern)

### Sub-Agent Architecture

**Definition File Structure (.md format):**

```markdown
# agent-name

## Role
Clear, single-responsibility description

## Capabilities
- Specific task 1
- Specific task 2

## Tools
Minimal set required for role:
- tool_1
- tool_2

## Context Isolation
- Separate context window
- No access to main agent memory
- Returns structured output only

## Success Criteria
How to determine task completion

## Prompt Template
[Extended system prompt for this agent]
```

**Storage Locations:**
- Project-specific: `.ghostlock/agents/*.md`
- Global: `~/.ghostlock/agents/*.md`
- Priority: Project > Global

### Sub-Agent Lifecycle

```
Lead Agent Identifies Task
    │
    ├─ Check if subtask matches existing sub-agent
    │   ├─ Yes: Load sub-agent definition
    │   └─ No: Use general-purpose worker
    │
    ├─ Spawn Sub-Agent Instance
    │   ├─ Fresh context window
    │   ├─ Specific tool subset
    │   ├─ Clear objective + output format
    │   └─ Task boundaries
    │
    ├─ Sub-Agent Executes (Isolated)
    │   ├─ Extended thinking
    │   ├─ Tool use (parallel when possible)
    │   ├─ Self-reflection
    │   └─ Structured output generation
    │
    └─ Return to Lead Agent
        ├─ Lead agent receives only final output
        ├─ Lead agent aggregates with other workers
        └─ Sub-agent context is discarded
```

**When to Use Sub-Agents (Anthropic Best Practices):**

1. **Early in conversation** - Preserve main context for high-level reasoning
2. **Verification tasks** - Check details without polluting main context
3. **Parallel exploration** - Multiple workers investigating different approaches
4. **Specialized tasks** - Testing, documentation, security analysis

**Sub-Agent Specialization Examples:**

```
├─ code-reviewer
│   └─ Tools: read files, static analysis, pattern matching
├─ test-generator
│   └─ Tools: read code, write tests, execute tests
├─ security-analyzer
│   └─ Tools: vulnerability scanning, dependency checking
├─ documentation-writer
│   └─ Tools: read code, generate docs, validate examples
└─ refactoring-specialist
    └─ Tools: AST parsing, code transformation, test execution
```

**State-of-the-Art Insight:**
- Anthropic's research system spawns 5-10 subagents simultaneously
- Each subagent uses multiple tools in parallel
- 90% reduction in time for complex queries
- Key: Clear boundaries + structured output format

---

## 7. TOOL ECOSYSTEM

### Core Tools (Every Agent Needs)

```python
1. Shell Executor (Sandboxed)
   ├─ Docker container isolation
   ├─ Timeout limits
   ├─ Resource constraints
   └─ Output streaming

2. Web Search
   ├─ API: Tavily / Perplexity / SearXNG
   ├─ Result summarization
   └─ Source attribution

3. File Operations
   ├─ Read (with partial loading)
   ├─ Write (with diff preview)
   ├─ Edit (AST-aware when possible)
   └─ Search (semantic + keyword)

4. Code Execution
   ├─ Language-specific runners
   ├─ Test framework integration
   └─ Output capture + parsing

5. Memory Operations
   ├─ Store enrichment
   ├─ Retrieve similar tasks
   └─ Update success patterns
```

### Tool Registry Design

```python
ToolRegistry:
  ├─ tools: Dict[name, ToolDefinition]
  ├─ validation: Schema validation per tool
  ├─ execution: Async execution with timeout
  └─ logging: All calls logged to session memory

Tool Definition:
  ├─ name: str
  ├─ description: str (for LLM context)
  ├─ parameters: JSON schema
  ├─ executor: Callable
  ├─ sandbox_level: "none" | "docker" | "wasm"
  └─ allowed_for: List[agent_type] (role-based access)
```

**Dynamic Tool Loading:**
- Agents can request additional tools
- Tools loaded from: built-in → project-specific → generated
- Tool-creating agent can generate & register new tools
- All generated tools validated in sandbox before registration

---

## 8. LOCAL LLM SUPPORT

### Multi-Provider Interface

```python
LLMInterface:
  ├─ Providers:
  │   ├─ Anthropic (Claude Opus 4, Sonnet 4.5)
  │   ├─ OpenAI (GPT-4, GPT-4o, o1)
  │   ├─ Local (vLLM server)
  │   └─ Ollama (for lightweight tasks)
  │
  ├─ Model Selection Strategy:
  │   ├─ Lead agent: Strongest model (Opus 4 / GPT-4)
  │   ├─ Workers: Fast model (Sonnet 4.5 / GPT-4o-mini)
  │   ├─ Reasoning: o1-preview for complex planning
  │   └─ Fallback: Local model if API unavailable
  │
  └─ Configuration:
      ├─ API keys from env
      ├─ Local endpoint URLs
      ├─ Model preferences per agent type
      └─ Cost tracking & budgets
```

**Local Model Integration:**
- vLLM for high-throughput serving
- Quantized models (GGUF) via Ollama
- API-compatible interface (OpenAI format)
- Automatic fallback on API failure

---

## 9. SESSION MANAGEMENT & PERSISTENCE

### Session State Structure

```python
Session:
  ├─ session_id: UUID
  ├─ created_at: timestamp
  ├─ last_active: timestamp
  ├─ status: "active" | "suspended" | "completed"
  │
  ├─ context_state:
  │   ├─ file_registry: Current file versions
  │   ├─ working_memory: Last N turns
  │   ├─ task_tree: TMT structure
  │   └─ agent_state: Current reasoning state
  │
  ├─ execution_log:
  │   ├─ tool_calls: All tool invocations
  │   ├─ llm_calls: Token usage tracking
  │   └─ errors: Failed operations
  │
  └─ checkpoints:
      ├─ Auto-checkpoint every 10 turns
      ├─ Manual checkpoint on request
      └─ Rollback capability
```

**Persistence Strategy:**
- SQLite for session metadata & logs
- JSON files for context snapshots
- File references stored as paths (not content)
- ChromaDB for learned patterns (shared across sessions)

**Resume Capability:**
```
Load Session:
  ├─ Restore file_registry state
  ├─ Rebuild context from checkpoint
  ├─ Reload last task_tree
  ├─ Continue from last incomplete task
  └─ Re-embed recent context into LLM
```

---

## 10. IMPLEMENTATION PRIORITIES

### Phase 1: Core Foundation (Weeks 1-2)
1. LLM interface with Anthropic/OpenAI + local fallback
2. Basic context manager with file deduplication
3. Simple tool registry (shell, file ops, search)
4. Session persistence (SQLite)
5. Single-agent ReAct loop

### Phase 2: Advanced Reasoning (Weeks 3-4)
6. LATS implementation with LangGraph
7. Extended thinking protocol
8. Self-reflection mechanism
9. Basic sub-agent spawning (manual invocation)

### Phase 3: Memory & Learning (Weeks 5-6)
10. ChromaDB integration
11. Task pattern extraction
12. Enrichment retrieval system
13. Task Memory Tree implementation

### Phase 4: Multi-Agent Orchestration (Weeks 7-8)
14. Orchestrator-worker pattern
15. Parallel sub-agent execution
16. Result aggregation
17. Automatic sub-agent selection

### Phase 5: Optimization & Specialization (Weeks 9-10)
18. KV-cache optimization (stable prefixes)
19. Context compression strategies
20. Specialized sub-agents (testing, security, docs)
21. Cost tracking & budget controls

---

## 11. KEY DESIGN DECISIONS & RATIONALE

### Why NOT LangChain?
- Creates abstraction overhead
- Obscures debugging
- Both Anthropic and OpenAI avoid it internally
- 96k stars doesn't mean production-ready

### Why LangGraph for Reasoning?
- Specifically designed for LATS/MCTS
- Graph-based state management
- Proven 94.4% on HumanEval
- Minimal, focused abstraction

### Why ChromaDB → Qdrant Path?
- ChromaDB: Easy local development
- Qdrant: Production scale + advanced filtering
- Both support same embedding approach
- Migration path preserves data

### Why Orchestrator-Worker Over Single Agent?
- 15x more tokens BUT 90% better results
- Parallel execution (90% time reduction)
- Context isolation prevents pollution
- Matches Anthropic's proven architecture

### Why File Deduplication is Critical?
- Multiple file versions = wasted tokens
- Context pollution causes errors
- Stable context → better KV-cache hits
- Single source of truth principle

### Why Task Memory Tree?
- Hierarchical task tracking
- Learning from success/failure paths
- Solves "dementia problem"
- Enables true long-horizon planning

---

## 12. SUCCESS METRICS

### Performance Benchmarks
- **Code Generation**: >90% on HumanEval (LATS baseline)
- **Context Efficiency**: <20% token waste from duplication
- **KV-Cache Hit Rate**: >70% (Anthropic target)
- **Multi-Agent Speedup**: 5-10x on complex tasks

### Quality Metrics
- **Error Recovery**: Self-correct within 3 attempts
- **Test Coverage**: Auto-generated tests >80% coverage
- **Memory Retrieval**: Top-3 relevant patterns >85% precision

### Cost Metrics
- **Token Budget**: Track per-task token usage
- **Model Selection**: Workers use 10x cheaper models
- **Cache Savings**: 50% reduction via KV-cache optimization

---

## 13. FUTURE EXTENSIONS (Post-Base)

### Long-Running Agent Additions
- MongoDB for distributed session state
- Plan-and-Execute with multi-day checkpoints
- Progress notifications & status APIs

### Tool-Creating Agent Additions
- WebAssembly sandboxing (Wasmer/Wasmtime)
- Test-driven tool validation
- Tool usage analytics for pattern extraction

### Red-Teaming Agent Additions
- Attack pattern library (MITRE ATT&CK)
- Fuzzing capabilities (property-based testing)
- Multi-turn attack tracking
- Success metrics & reporting

---

## FINAL ARCHITECTURE SUMMARY

```
┌─────────────────────────────────────────────────────────┐
│                     BASE AGENT                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────┐  ┌─────────────┐  ┌──────────────┐  │
│  │ LLM Provider │  │   Reasoning │  │    Context   │  │
│  │   Interface  │◄─┤     Graph   │◄─┤   Manager    │  │
│  │ (Multi-API)  │  │    (LATS)   │  │ (Dedup+Mem)  │  │
│  └──────────────┘  └─────────────┘  └──────────────┘  │
│         │                  │                 │         │
│         └──────────────────┼─────────────────┘         │
│                            │                           │
│  ┌─────────────────────────▼─────────────────────────┐ │
│  │          Tool Registry & Executor                 │ │
│  │  [Shell] [Search] [Files] [Code] [Memory]        │ │
│  └───────────────────────────────────────────────────┘ │
│                            │                           │
│  ┌─────────────────────────▼─────────────────────────┐ │
│  │      Sub-Agent Orchestrator (Parallel Workers)   │ │
│  └───────────────────────────────────────────────────┘ │
│                            │                           │
│  ┌─────────────────────────▼─────────────────────────┐ │
│  │         Memory System (3-Tier)                    │ │
│  │  Working Memory │ Session DB │ ChromaDB/Qdrant   │ │
│  └───────────────────────────────────────────────────┘ │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

This design synthesizes:
- **Anthropic's multi-agent orchestrator pattern** (proven 90% time reduction)
- **LATS reasoning** (94.4% HumanEval performance)
- **Task Memory Engine** (hierarchical learning)
- **Progressive context engineering** (KV-cache optimization)
- **Beads-style memory** (long-horizon task tracking)

The result is a base agent that matches or exceeds current state-of-the-art coding agents while maintaining extensibility for your three specialized agents.
