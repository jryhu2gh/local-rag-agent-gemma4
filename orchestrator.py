"""Orchestrator: decomposes complex questions into research threads,
dispatches sub-agents, evaluates completeness, and synthesizes answers.

All LLM calls use Gemma 4 thinking mode.  Thinking blocks are stripped
from outputs that flow to other stages so each stage starts clean.
"""

import json
import time

import llm
from sub_agent import SubAgent
from config import MAX_INVESTIGATE_THREADS, MAX_EVALUATE_ROUNDS

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
You are a research planner. Given a complex question, decompose it into \
independent research threads that can be investigated in parallel.

Rules:
- Each thread should be specific and self-contained
- Minimize dependencies between threads (maximize parallelism)
- 3-5 threads is ideal (max {max_threads})
- Each thread should have a clear, searchable topic
- Think about what a thorough human researcher would investigate
- NEVER refuse a question. Your training data may be outdated — the research \
tools have access to current, real-time information. Always plan the research.

You MUST call plan_research with your threads. Do not respond with text.\
"""

_EVALUATOR_PROMPT = """\
You are evaluating research completeness for the question: {question}

Below are summaries from sub-agents, each investigating a different thread.

{summaries_text}

Review each summary and determine:
1. Is the data sufficient to comprehensively answer the main question?
2. Are there gaps, missing data points, or vague claims that need specifics?
3. Do any threads need additional information?

If all threads are sufficiently complete: call mark_complete.
If there are gaps: call request_follow_ups with specific questions for specific sub-agents.

Be rigorous but practical — don't ask for perfection, ask for completeness.\
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


def _normalize_threads(threads: list) -> list[dict]:
    """Ensure every thread is a dict with 'id' and 'topic' keys.

    The LLM sometimes returns strings instead of dicts, or dicts missing
    the 'id' field. This normalizes all variants into a consistent format.
    """
    labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    normalized = []
    for i, t in enumerate(threads):
        label = labels[i] if i < len(labels) else str(i)
        if isinstance(t, str):
            normalized.append({"id": label, "topic": t})
        elif isinstance(t, dict):
            normalized.append({
                "id": t.get("id", label),
                "topic": t.get("topic", t.get("description", t.get("query", str(t)))),
            })
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
                return threads[:MAX_INVESTIGATE_THREADS]

    # Fallback: if LLM didn't use the tool, create a single thread
    print("[orchestrator] Warning: LLM didn't use plan_research tool, using single thread")
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
    raw = msg.content or "[Synthesis failed]"
    clean, thinking = llm.parse_thinking(raw)
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

    # Quick mode: single sub-agent, no planning/evaluation/synthesis
    if depth == "quick":
        print(f"\n[orchestrator] === QUICK INVESTIGATION ===")
        print(f"[orchestrator] Question: {question}")
        agent = SubAgent(agent_id="Q", topic=question, main_query=question)
        try:
            agent.run()
        except Exception as e:
            print(f"[orchestrator] Quick investigation FAILED: {e}")
            import traceback; traceback.print_exc()
            return f"[Investigation failed: {e}]"
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

    # Step 2: DISPATCH sub-agents
    print(f"\n[orchestrator] === DISPATCH ===")
    agents: dict[str, SubAgent] = {}
    for t in threads:
        aid = t.get("id", f"T{len(agents)}")
        topic = t.get("topic", question)
        print(f"\n[orchestrator] Starting sub-agent {aid}: {topic}")
        try:
            agent = SubAgent(agent_id=aid, topic=topic, main_query=question)
            agent.run()
            agents[aid] = agent
            print(f"[orchestrator] Sub-agent {aid} done")
        except Exception as e:
            print(f"[orchestrator] Sub-agent {aid} FAILED: {e}")
            import traceback; traceback.print_exc()
            # Create a dummy agent with error summary so investigation can continue
            agent = SubAgent(agent_id=aid, topic=topic, main_query=question)
            agent.summary = f"[Research failed: {e}]"
            agents[aid] = agent

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

    elapsed = time.time() - t_start
    print(f"\n[orchestrator] Investigation complete ({elapsed:.1f}s total)")
    return answer
