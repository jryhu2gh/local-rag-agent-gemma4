"""Investigate skill — multi-agent research for complex questions."""

import orchestrator

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
    return orchestrator.investigate(query, depth=depth)
