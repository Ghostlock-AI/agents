# ReWOO Strategy Fixes

## Issues Identified

### Issue 1: Plan Created But No Synthesis
**Symptom:** After creating a plan and executing tools, the workflow stopped without synthesizing a final answer.

**Example:**
```
User: "when was the last time trump was in Australia"
Output:
[PLAN CREATED]
{ "steps": [...] }
[TOOL: ddgs_search] [TOOL: ddgs_search]
(no final answer)
```

**Root Cause:** The routing logic after `plan_to_calls` was checking the plan state rather than checking if tool_calls were actually created. This caused premature routing to the synthesizer before tools executed.

### Issue 2: OpenAI API Error on Second Query
**Symptom:** Second query in the same session caused OpenAI API error about unresolved tool_calls.

**Example:**
```
Error: Error code: 400 - {'error': {'message': "An assistant message with
'tool_calls' must be followed by tool messages responding to each
'tool_call_id'. The following tool_call_ids did not have response messages..."}}
```

**Root Cause:**
1. AIMessages with `tool_calls` remained in conversation history
2. When the synthesizer invoked the LLM with these messages, OpenAI expected ToolMessage responses
3. The synthesizer was using `llm_with_tools` which still had tools bound, allowing it to create new tool_calls

## Solutions Implemented

### Fix 1: Improved Routing Logic
Changed `after_plan_route` to check for actual tool_calls in messages rather than inspecting plan state.

**Before:**
```python
def after_plan_route(state) -> Literal["tools", "synthesizer"]:
    plan = state.get("plan") or {}
    steps: List[Dict[str, Any]] = plan.get("steps", [])
    executed: Set[str] = set(state.get("executed") or [])
    for s in steps:
        sid = str(s.get("id"))
        if sid in executed:
            continue
        deps = s.get("depends_on", []) or []
        if all(str(d) in executed for d in deps):
            return "tools"
    return "synthesizer"
```

**After:**
```python
def after_plan_route(state) -> Literal["tools", "synthesizer"]:
    """Route after plan_to_calls: if we created tool_calls, go to tools; else synthesize."""
    messages = state.get("messages", [])
    # Check if the last message has tool_calls
    if messages:
        last_msg = messages[-1]
        if isinstance(last_msg, AIMessage) and hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
            return "tools"
    return "synthesizer"
```

### Fix 2: Clean Message History for Synthesis
Added message filtering to remove AIMessages with unresolved tool_calls before synthesis.

**Before:**
```python
def synthesizer_node(state):
    messages = state["messages"]
    # ... build synthesis_prompt ...
    response = llm_with_tools.invoke(messages + [SystemMessage(content=synthesis_prompt)])
    return {"messages": [response]}
```

**After:**
```python
def synthesizer_node(state):
    messages = state["messages"]
    results: List[Dict[str, Any]] = list(state.get("results") or [])

    # Filter messages to only include HumanMessage, SystemMessage, and AIMessage without tool_calls
    clean_messages = []
    for msg in messages:
        if isinstance(msg, (HumanMessage, SystemMessage)):
            clean_messages.append(msg)
        elif isinstance(msg, AIMessage):
            # Only include AIMessage if it doesn't have unresolved tool_calls
            if not (hasattr(msg, "tool_calls") and msg.tool_calls):
                clean_messages.append(msg)

    # ... build synthesis_prompt ...

    # Use LLM without tools for synthesis
    llm_for_synthesis = llm_with_tools.bind(tools=[])
    response = llm_for_synthesis.invoke(clean_messages + [SystemMessage(content=synthesis_prompt)])
    return {"messages": [response]}
```

### Fix 3: Remove Tools from Planning and Synthesis
Both planner and synthesizer now use `llm_with_tools.bind(tools=[])` to prevent accidental tool_calls.

**Planner:**
```python
# Use a fresh LLM without tools for planning to avoid tool_calls
llm_for_planning = llm_with_tools.bind(tools=[])
response = llm_for_planning.invoke(prompt)
```

**Synthesizer:**
```python
# Use LLM without tools for synthesis
llm_for_synthesis = llm_with_tools.bind(tools=[])
response = llm_for_synthesis.invoke(clean_messages + [SystemMessage(content=synthesis_prompt)])
```

## Expected Behavior After Fixes

### Workflow Flow
1. **Planner** → Creates plan (no tool_calls)
2. **Plan-to-Calls** → Expands ready steps to tool_calls
3. **Tools** → Executes tool_calls
4. **Collect** → Gathers results, marks steps as executed
5. **Loop or Synthesize:**
   - If more steps remain → back to Plan-to-Calls
   - If all steps done → to Synthesizer
6. **Synthesizer** → Creates final answer from results

### Example Success Case

**Query:** "What day is today?"

**Expected Output:**
```
[PLAN CREATED]
{ "steps": [{"id": 1, "tool": "shell_exec", "args": {"command": "date +%A"}, "depends_on": []}] }

[TOOL: shell_exec]

Today is Monday, October 27, 2025.
```

**Flow:**
1. Planner creates plan with 1 step
2. Plan-to-calls creates tool_call for shell_exec
3. Tools execute shell_exec
4. Collect marks step 1 as executed
5. After_tools_route sees all steps done → synthesizer
6. Synthesizer combines results into final answer

## Testing Recommendations

1. **Simple queries** (1 step): "What day is today?"
2. **Multi-step queries** (dependent steps): "Search for React docs and fetch the first result"
3. **Parallel queries** (independent steps): "Get weather in SF and NYC"
4. **Multiple queries in session** to verify no cross-contamination

## Additional Issues and Fixes (After Initial Testing)

### Issue 3: Synthesizer Changing Tool Results
**Symptom:** ReWOO executed tools correctly but synthesizer changed the results.

**Example:**
```
Tool output: "Monday, October 27, 2025"
Synthesizer output: "Today is Monday, October 23, 2023" ❌
```

**Root Cause:** The synthesis prompt didn't emphasize using ONLY tool results. The synthesizer was mixing tool outputs with training data.

**Fix:** Enhanced synthesis prompt with explicit instructions (lines 217-225):
```python
synthesis_prompt = (
    "CRITICAL: You must synthesize your answer using ONLY the tool execution results below. "
    "Do NOT use any information from your training data. "
    "Do NOT modify dates, times, or facts from the tool outputs. "
    "Simply present the information from the tools in a clear, concise way.\n\n"
    "Tool execution results:\n"
    + "\n".join(lines)
    + "\n\nProvide a direct answer based ONLY on the above results."
)
```

### Issue 4: Planner Creating Overly Complex Plans
**Symptom:** Planner created redundant 4-step plans for simple queries like "What time is it?"

**Root Cause:** Planning prompt examples were too detailed and confusing the LLM.

**Fix:** Simplified planning guidance (lines 84-90):
- Removed specific examples that were causing confusion
- Added "Keep plans simple - don't create redundant steps"
- Clearer rules: "For current date/time: Use shell_exec with date command"

## Final Test Results

**Before All Fixes:**
```
Query: "What day is today?"
Output: "Today is Monday, October 23, 2023" ❌ (wrong date from training data)
```

**After All Fixes:**
```
Query: "What day is today?"
Output: "Today is Monday, October 27, 2025" ✅ (correct current date)

Query: "What time is it?"
Output: "The current time is 16:11:25 GMT on October 27, 2025" ✅ (correct time)
```

## Summary of All Fixes

1. ✅ **Routing Logic** - Check actual tool_calls instead of plan state
2. ✅ **Message Filtering** - Remove AIMessages with unresolved tool_calls before synthesis
3. ✅ **Tool-Free LLMs** - Use `bind(tools=[])` for planner and synthesizer
4. ✅ **Synthesis Prompt** - Emphasize using ONLY tool results, no training data
5. ✅ **Planning Guidance** - Simplified rules to prevent overly complex plans

## Remaining Considerations

1. **Error Handling:** If a tool fails, the workflow should still synthesize with available results
2. **Empty Results:** If all tools return no useful data, synthesizer should indicate this
3. **Token Limits:** Very long results might exceed context limits
4. **Streaming:** Consider streaming tool results instead of batching for better UX

## Files Modified

- `src/reasoning/strategies/rewoo.py`:
  - Line 125: Added `llm_for_planning` without tools
  - Lines 84-90: Simplified planning rules to prevent complex plans
  - Lines 217-225: Enhanced synthesis prompt to prevent hallucination
  - Line 233: Rewrote `after_plan_route` to check actual tool_calls
  - Lines 199-222: Enhanced `synthesizer_node` with message filtering and tool-free LLM
