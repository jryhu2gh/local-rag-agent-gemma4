"""Reflect skill — check what the agent already knows.

Searches local history and indexed documents, then uses the LLM to
summarize what is known and identify what is missing.
"""

import json

import history
import llm
import toolkit

DEFINITION = {
    "type": "function",
    "function": {
        "name": "reflect",
        "description": (
            "Check what you already know about a topic. Searches past conversations "
            "and locally indexed documents, then summarizes your existing knowledge "
            "and identifies gaps. Use this FIRST before looking things up online."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to check your knowledge about",
                }
            },
            "required": ["query"],
        },
    },
}

_REFLECT_PROMPT = """\
You are reviewing what is already known about a topic from local sources.

The user's question: {query}

Below are results from past conversations and indexed documents.

## From past conversations:
{history_section}

## From indexed documents:
{documents_section}

Provide a brief assessment:
1. **What I know**: Summarize ONLY facts that DIRECTLY answer the question. Ignore tangentially related results.
2. **What I don't know**: What information is missing to answer the question?
3. **Can the question be answered?**: Reply YES or NO. Say YES only if the sources above contain a direct, substantive answer. If the sources only mention the topic in passing, or contain no real answer, say NO.

Be concise and specific. Be strict — having related text is NOT the same as having an answer.\
"""


def execute(query: str) -> str:
    """Search local knowledge and summarize what is known vs unknown."""
    print(f"  [reflect] Searching local knowledge for: {query!r}")

    # Tier 1: Check past conversations
    history_results = history.search(query)
    if history_results:
        history_section = "\n".join(
            f"- [{r.get('timestamp', '?')}] {r.get('text', '')[:200]}"
            for r in history_results
        )
        print(f"  [reflect] Found {len(history_results)} history matches")
    else:
        history_section = "(No relevant past conversations found)"
        print(f"  [reflect] No history matches")

    # Tier 2: Search indexed documents
    doc_results_raw = toolkit.search_documents(query)
    doc_results = json.loads(doc_results_raw).get("results", [])
    if doc_results:
        documents_section = "\n".join(
            f"- [{r.get('source_file', '?')}] {r.get('text', '')[:200]}"
            for r in doc_results
        )
        print(f"  [reflect] Found {len(doc_results)} document matches")
    else:
        documents_section = "(No relevant documents found)"
        print(f"  [reflect] No document matches")

    # If nothing found locally, skip LLM summarization
    if not history_results and not doc_results:
        print(f"  [reflect] No local knowledge found")
        return json.dumps({
            "query": query,
            "status": "no_local_knowledge",
            "summary": "No relevant information found in past conversations or local documents.",
            "recommendation": "Use investigate to search the web for this information.",
        }, indent=2)

    # Use LLM to summarize what we know vs don't know
    print(f"  [reflect] Summarizing with LLM...")
    messages = [
        {"role": "system", "content": _REFLECT_PROMPT.format(
            query=query,
            history_section=history_section,
            documents_section=documents_section,
        )},
        {"role": "user", "content": f"What do we know about: {query}"},
    ]

    msg = llm.call(messages, max_tokens=1024)
    summary = msg.content or "[Summarization failed]"
    print(f"  [reflect] Summary ready ({len(summary)} chars)")

    # Check if the LLM concluded it can answer the question
    summary_lower = summary.lower()
    can_answer = "can the question be answered?: yes" in summary_lower or \
                 "can the question be answered?**: yes" in summary_lower

    if can_answer:
        status = "answered"
        recommendation = "Local knowledge is sufficient to answer this question."
    else:
        status = "insufficient"
        recommendation = "Local knowledge is NOT sufficient. You MUST call investigate to search the web for this information. Do NOT answer from local knowledge alone."

    print(f"  [reflect] Status: {status}")

    return json.dumps({
        "query": query,
        "status": status,
        "summary": summary,
        "recommendation": recommendation,
        "sources": {
            "history_matches": len(history_results),
            "document_matches": len(doc_results),
        },
    }, indent=2)
