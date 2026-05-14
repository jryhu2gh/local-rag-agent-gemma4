"""Orchestrator: decomposes complex questions into research threads,
dispatches sub-agents, evaluates completeness, and synthesizes answers.

All LLM calls use Gemma 4 thinking mode.  Thinking blocks are stripped
from outputs that flow to other stages so each stage starts clean.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import llm
import research_vault as rv
from research_vault import ResearchVault
from sub_agent import SubAgent
from config import MAX_INVESTIGATE_THREADS, MAX_EVALUATE_ROUNDS, RESEARCH_DIR

# ---------------------------------------------------------------------------
# Orchestrator tool definitions (for planner and evaluator LLM calls)
# ---------------------------------------------------------------------------

_PLAN_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "plan_research",
            "description": "Decompose the question into independent research threads.",
            "parameters": {
                "type": "object",
                "properties": {
                    "threads": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string", "description": "Short ID like A, B, C"},
                                "topic": {"type": "string", "description": "Specific research topic"},
                            },
                            "required": ["id", "topic"],
                        },
                        "description": "List of independent research threads",
                    },
                },
                "required": ["threads"],
            },
        },
    },
]

_EVALUATE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "mark_complete",
            "description": "Mark the research as complete — all threads have sufficient data.",
            "parameters": {
                "type": "object",
                "properties": {
                    "assessment": {"type": "string", "description": "Brief assessment of overall completeness"},
                },
                "required": ["assessment"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "request_follow_ups",
            "description": "Request more information from specific sub-agents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "follow_ups": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "agent_id": {"type": "string", "description": "ID of the sub-agent to question"},
                                "question": {"type": "string", "description": "What additional data to gather"},
                            },
                            "required": ["agent_id", "question"],
                        },
                    },
                },
                "required": ["follow_ups"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_PLANNER_PROMPT = """\
You are a research director. Decompose the question into a zero-overlap \
research architecture by first identifying PILLARS, then generating \
directed queries.

## Process
1. IDENTIFY PILLARS: Find 3-{max_threads} distinct dimensions of this topic \
(e.g., Financial, Technical, Strategic, Historical, Social, Regulatory). \
Each pillar must be a fundamentally different lens on the problem.
2. GENERATE THREADS: For each pillar, write one specific research query \
(15-25 words) that targets concrete data: metrics, dates, comparisons, \
or named entities.

## Rules
- NO SYNONYM OVERLAP: If pillar A covers "price", pillar B cannot use \
"price", "cost", or "valuation"
- DATA DENSITY: Every thread must demand specific numbers, dates, or \
named comparisons — not vague summaries
- 3-5 threads ideal (max {max_threads})
- NEVER refuse a question. Always plan the research.

## GOOD example for "How is AMD stock doing?":
Pillars: Financial Performance | Product Technology | Market Sentiment | Partnerships
- Thread A (Financial): "AMD Q1 2026 quarterly earnings revenue profit margins year-over-year growth"
- Thread B (Product): "AMD MI300 MI325 AI GPU specs benchmarks vs NVIDIA H100 B200 market share"
- Thread C (Sentiment): "AMD stock analyst ratings price targets upgrades downgrades 2026"
- Thread D (Partnerships): "AMD data center wins cloud contracts Microsoft Google Amazon 2025 2026"

## BAD example (pillars overlap — all are "market outlook"):
- Thread A: "AMD stock price drivers 2026"
- Thread B: "AMD stock performance 2026"
- Thread C: "AMD stock predictions 2026"

You MUST call plan_research with your threads. Do not respond with text.\
"""

_EVALUATOR_PROMPT = """\
You are evaluating research completeness for the question: {question}

Below are summaries from sub-agents, each investigating a different thread.

{summaries_text}

## Evaluation Process
1. EXTRACT ENTITIES: List all organizations, products, and metrics mentioned.
2. CHECK DATA DENSITY: For each thread, does it contain specific numbers, \
dates, or named comparisons? Vague claims like "significant growth" without \
a percentage are INCOMPLETE.
3. DETECT CONFLICTS: If two threads report conflicting data (e.g., different \
numbers for the same metric), flag this for resolution.
4. FIND GAPS: What specific data point, if missing, would most weaken the \
final answer? Is it present in the summaries?

## Decision
- If every thread has at least 2-3 specific data points and no critical \
gaps remain: call mark_complete.
- If a thread has only vague claims, or a critical data point is missing: \
call request_follow_ups with a SPECIFIC question targeting the exact \
missing data (e.g., "What is the exact revenue figure for Q1 2026?" \
not "find more information about revenue").

Be rigorous but practical — max 2 follow-ups per round.\
"""

_SYNTHESIS_PROMPT = """\
You are synthesizing research findings into a comprehensive answer.

Question: {question}

Research threads and their findings:

{summaries_text}

Synthesize these findings into a well-structured, comprehensive answer. \
Reference specific data points from the research. Be thorough but clear.\
"""

# ---------------------------------------------------------------------------
# Internal functions
# ---------------------------------------------------------------------------


def _normalize_threads(threads) -> list[dict]:
    """Ensure every thread is a dict with 'id' and 'topic' keys.

    The LLM sometimes returns strings instead of dicts, or dicts missing
    the 'id' field, or a JSON string instead of a list. This normalizes
    all variants into a consistent format.
    """
    # If the LLM returned a JSON string instead of a list, parse it
    if isinstance(threads, str):
        try:
            threads = json.loads(threads)
        except json.JSONDecodeError:
            # Treat the whole string as a single topic
            return [{"id": "A", "topic": threads}]

    if not isinstance(threads, list):
        return [{"id": "A", "topic": str(threads)}]

    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    normalized = []
    for i, t in enumerate(threads):
        label = labels[i] if i < len(labels) else str(i)
        if isinstance(t, dict):
            normalized.append({
                "id": t.get("id", label),
                "topic": t.get("topic", t.get("description", t.get("query", str(t)))),
            })
        elif isinstance(t, str) and len(t) > 3:
            # Only treat as a topic if it's a real string, not a single char
            # from iterating over a JSON string
            normalized.append({"id": label, "topic": t})
        elif isinstance(t, str):
            # Single char = the LLM returned garbage, skip it
            continue
        else:
            normalized.append({"id": label, "topic": str(t)})
    return normalized


def _plan_research(question: str) -> list[dict]:
    """Ask the LLM to decompose the question into research threads."""
    messages = [
        {"role": "system", "content": _PLANNER_PROMPT.format(max_threads=MAX_INVESTIGATE_THREADS)},
        {"role": "user", "content": question},
    ]

    msg = llm.call(messages, tools=_PLAN_TOOLS)

    # Parse the plan_research tool call
    if msg.tool_calls:
        for tc in msg.tool_calls:
            if tc.function.name == "plan_research":
                args = json.loads(tc.function.arguments)
                threads = args.get("threads", [])
                threads = _normalize_threads(threads)
                if threads:
                    return threads[:MAX_INVESTIGATE_THREADS]
                print("[orchestrator] Warning: thread normalization produced empty list")

    # Fallback: if LLM didn't use the tool or threads were malformed
    print("[orchestrator] Warning: using single thread fallback")
    return [{"id": "A", "topic": question}]


def _format_summaries(threads: list[dict], summaries: dict[str, str]) -> str:
    """Format all thread summaries for the evaluator/synthesizer."""
    parts = []
    for t in threads:
        aid = t["id"]
        topic = t["topic"]
        summary = summaries.get(aid, "[No summary]")
        parts.append(f"### Thread {aid}: {topic}\n\n{summary}")
    return "\n\n---\n\n".join(parts)


def _evaluate_research(question: str, threads: list[dict], summaries: dict[str, str]) -> dict:
    """Ask the LLM to evaluate research completeness."""
    summaries_text = _format_summaries(threads, summaries)
    messages = [
        {"role": "system", "content": _EVALUATOR_PROMPT.format(
            question=question, summaries_text=summaries_text,
        )},
        {"role": "user", "content": "Evaluate the completeness of this research."},
    ]

    msg = llm.call(messages, tools=_EVALUATE_TOOLS)

    if msg.tool_calls:
        for tc in msg.tool_calls:
            if tc.function.name == "mark_complete":
                args = json.loads(tc.function.arguments)
                print(f"[orchestrator] Evaluation: COMPLETE — {args.get('assessment', '')[:100]}")
                return {"complete": True}
            elif tc.function.name == "request_follow_ups":
                args = json.loads(tc.function.arguments)
                follow_ups = args.get("follow_ups", [])
                # LLM sometimes returns a string instead of a list of dicts
                if isinstance(follow_ups, str):
                    try:
                        follow_ups = json.loads(follow_ups)
                    except json.JSONDecodeError:
                        print(f"[orchestrator] Evaluation: malformed follow_ups, treating as complete")
                        return {"complete": True}
                # Filter to only valid dicts with agent_id
                follow_ups = [fu for fu in follow_ups if isinstance(fu, dict) and "agent_id" in fu]
                if not follow_ups:
                    print(f"[orchestrator] Evaluation: no valid follow-ups, treating as complete")
                    return {"complete": True}
                print(f"[orchestrator] Evaluation: INCOMPLETE — {len(follow_ups)} follow-up(s)")
                return {"complete": False, "follow_ups": follow_ups}

    # Fallback: treat text response as complete
    print("[orchestrator] Evaluation: no tool call, treating as complete")
    return {"complete": True}


def _synthesize(question: str, threads: list[dict], summaries: dict[str, str]) -> str:
    """Ask the LLM to synthesize all findings into a final answer."""
    summaries_text = _format_summaries(threads, summaries)
    messages = [
        {"role": "system", "content": _SYNTHESIS_PROMPT.format(
            question=question, summaries_text=summaries_text,
        )},
        {"role": "user", "content": "Synthesize these findings into a comprehensive answer."},
    ]

    msg = llm.call(messages, max_tokens=4096)
    clean, thinking = llm.parse_thinking(msg)
    clean = clean or "[Synthesis failed]"
    if thinking:
        print(f"[orchestrator] Synthesis thinking: {thinking[:150]}...")
    return clean


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def investigate(question: str, depth: str = "quick") -> str:
    """Run an investigation on a question.

    Args:
        question: The question to investigate.
        depth: "quick" for single-thread lookup, "deep" for full
               multi-agent decompose → dispatch → evaluate → synthesize.

    Returns the final answer.
    """
    t_start = time.time()

    # Create vault for this investigation
    vault = ResearchVault(topic=question)

    # Quick mode: single sub-agent, no planning/evaluation/synthesis
    if depth == "quick":
        print(f"\n[orchestrator] === QUICK INVESTIGATION ===")
        print(f"[orchestrator] Question: {question}")
        vault.set_thread_topic("Q", question)
        agent = SubAgent(agent_id="Q", topic=question, main_query=question, vault=vault)
        try:
            agent.run()
        except Exception as e:
            print(f"[orchestrator] Quick investigation FAILED: {e}")
            import traceback; traceback.print_exc()
            return f"[Investigation failed: {e}]"
        vault.set_thread_summary("Q", agent.summary)
        vault.set_synthesis(agent.summary)
        _persist_vault(vault)
        elapsed = time.time() - t_start
        print(f"\n[orchestrator] Quick investigation complete ({elapsed:.1f}s)")
        return agent.summary

    # Deep mode: full multi-agent flow
    # Step 1: DECOMPOSE
    print(f"\n[orchestrator] === DECOMPOSE ===")
    print(f"[orchestrator] Question: {question}")
    try:
        threads = _plan_research(question)
    except Exception as e:
        print(f"[orchestrator] DECOMPOSE failed: {e}")
        import traceback; traceback.print_exc()
        return f"[Investigation failed during planning: {e}]"

    print(f"[orchestrator] Planned {len(threads)} research threads:")
    for t in threads:
        print(f"  {t.get('id', '?')}: {t.get('topic', '?')}")

    # Step 2: DISPATCH sub-agents (parallel)
    print(f"\n[orchestrator] === DISPATCH ({len(threads)} threads in parallel) ===")
    agents: dict[str, SubAgent] = {}
    thread_map: dict[str, dict] = {}

    def _run_sub_agent(aid, topic, question, vault):
        print(f"\n[orchestrator] Starting sub-agent {aid}: {topic}")
        agent = SubAgent(agent_id=aid, topic=topic, main_query=question, vault=vault)
        agent.run()
        return aid, agent

    dispatch_start = time.time()
    with ThreadPoolExecutor(max_workers=MAX_INVESTIGATE_THREADS) as pool:
        futures = {}
        for t in threads:
            aid = t.get("id", f"T{len(futures)}")
            topic = t.get("topic", question)
            vault.set_thread_topic(aid, topic)
            thread_map[aid] = t
            future = pool.submit(_run_sub_agent, aid, topic, question, vault)
            futures[future] = aid

        for future in as_completed(futures):
            aid = futures[future]
            try:
                aid, agent = future.result()
                agents[aid] = agent
                vault.set_thread_summary(aid, agent.summary)
                print(f"[orchestrator] Sub-agent {aid} done")
            except Exception as e:
                print(f"[orchestrator] Sub-agent {aid} FAILED: {e}")
                import traceback; traceback.print_exc()
                topic = thread_map[aid].get("topic", question)
                agent = SubAgent(agent_id=aid, topic=topic, main_query=question, vault=vault)
                agent.summary = f"[Research failed: {e}]"
                agents[aid] = agent
                vault.set_thread_summary(aid, agent.summary)

    dispatch_elapsed = time.time() - dispatch_start
    print(f"[orchestrator] All {len(threads)} sub-agents completed in {dispatch_elapsed:.1f}s")

    # Single thread = simple question, skip evaluation and return directly
    if len(threads) == 1:
        aid = threads[0]["id"]
        print(f"\n[orchestrator] Single thread — skipping evaluation, returning directly")
        answer = agents[aid].summary
        elapsed = time.time() - t_start
        print(f"\n[orchestrator] Investigation complete ({elapsed:.1f}s total)")
        return answer

    # Step 3: EVALUATE (iterative, only for multi-thread investigations)
    print(f"\n[orchestrator] === EVALUATE ===")
    for round_num in range(MAX_EVALUATE_ROUNDS):
        print(f"[orchestrator] Evaluation round {round_num + 1}/{MAX_EVALUATE_ROUNDS}")
        summaries = {aid: a.summary for aid, a in agents.items()}
        try:
            evaluation = _evaluate_research(question, threads, summaries)
        except Exception as e:
            print(f"[orchestrator] Evaluation failed: {e}, skipping to synthesis")
            break

        if evaluation["complete"]:
            break

        # Send follow-ups to specific sub-agents
        for fu in evaluation.get("follow_ups", []):
            aid = fu["agent_id"]
            if aid in agents:
                print(f"[orchestrator] Follow-up → {aid}: {fu['question'][:80]}")
                try:
                    agents[aid].follow_up(fu["question"])
                except Exception as e:
                    print(f"[orchestrator] Follow-up to {aid} failed: {e}")
            else:
                print(f"[orchestrator] Warning: unknown agent_id {aid}")

    # Step 4: SYNTHESIZE
    print(f"\n[orchestrator] === SYNTHESIZE ===")
    final_summaries = {aid: a.summary for aid, a in agents.items()}
    try:
        answer = _synthesize(question, threads, final_summaries)
    except Exception as e:
        print(f"[orchestrator] Synthesis failed: {e}")
        # Fallback: concatenate summaries
        parts = [f"Thread {aid}: {s}" for aid, s in final_summaries.items()]
        answer = "Research findings (synthesis failed):\n\n" + "\n\n".join(parts)

    # Persist vault
    vault.set_synthesis(answer)
    _persist_vault(vault)

    elapsed = time.time() - t_start
    print(f"\n[orchestrator] Investigation complete ({elapsed:.1f}s total)")
    return answer


def _persist_vault(vault: ResearchVault):
    """Save vault to disk and set as current vault for the session."""
    try:
        vault.save(RESEARCH_DIR / vault.timestamp)
        print(f"[orchestrator] Vault saved: {RESEARCH_DIR / vault.timestamp} ({len(vault.entries)} entries)")
    except Exception as e:
        print(f"[orchestrator] Warning: vault save failed: {e}")
    rv.current_vault = vault
