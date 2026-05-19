"""Investigate skill — multi-agent research for complex questions."""

import json

import orchestrator
import research_vault as rv

DEFINITION = {
    "type": "function",
    "function": {
        "name": "investigate",
        "description": (
            "Run an investigation on a question. Use depth='quick' for simple lookups "
            "and depth='deep' for multi-threaded multi-agent research on complex questions "
            "(e.g., comparisons, analysis, strategic decisions)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The question to investigate",
                },
                "depth": {
                    "type": "string",
                    "enum": ["quick", "deep"],
                    "description": "Research depth: 'quick' for simple lookups, 'deep' for multi-agent research",
                },
            },
            "required": ["query"],
        },
    },
}


def execute(query: str, depth: str = "quick") -> str:
    """Run investigation at the specified depth."""
    synthesis = orchestrator.investigate(query, depth=depth)

    # Build source list from vault
    vault = rv.current_vault
    sources = []
    if vault:
        for entry in vault.entries.values():
            if entry["tool"] in ("web_search", "deep_research"):
                url = entry["args"].get("url", "")
                q = entry["args"].get("query", "")
                sources.append({"tool": entry["tool"], "query": q, "url": url})

    return json.dumps({
        "source": "web_investigation",
        "depth": depth,
        "synthesis": synthesis,
        "web_sources": sources,
    }, indent=2)
