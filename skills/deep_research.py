"""Deep research skill — search the web and read top results in full."""

import json

import toolkit

DEFINITION = {
    "type": "function",
    "function": {
        "name": "deep_research",
        "description": (
            "Search the web and read the top results in full. "
            "Use this for in-depth research on any topic when you need "
            "more than just search snippets. Returns full page content "
            "from multiple sources."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to research",
                },
                "max_sources": {
                    "type": "integer",
                    "description": "Number of pages to read in full (default 3, max 5)",
                },
            },
            "required": ["query"],
        },
    },
}


def execute(query: str, max_sources: int = 3) -> str:
    """Run deep web research."""
    result = toolkit.deep_research(query, max_sources)
    return json.dumps(result, indent=2)
