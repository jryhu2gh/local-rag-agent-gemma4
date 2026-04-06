"""Research skill — tiered local knowledge search.

Searches chat history first, then indexed documents, and returns
combined results so the LLM can synthesize a single answer.
"""

import json

import history
import toolkit

DEFINITION = {
    "type": "function",
    "function": {
        "name": "research",
        "description": (
            "Search local knowledge for information about a topic. "
            "Automatically checks past conversations first, then searches "
            "indexed documents and previously crawled web pages. "
            "Use this for any factual question."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for",
                }
            },
            "required": ["query"],
        },
    },
}


def execute(query: str) -> str:
    """Run tiered search: history → documents."""
    # Tier 1: Check past conversations
    history_results = history.search(query)

    # Tier 2: Search indexed documents
    doc_results_raw = toolkit.search_documents(query)
    doc_results = json.loads(doc_results_raw)

    return json.dumps({
        "from_history": history_results,
        "from_documents": doc_results.get("results", []),
    }, indent=2)
