"""Investigate skill — multi-agent research for complex questions."""

import orchestrator

DEFINITION = {
    "type": "function",
    "function": {
        "name": "investigate",
        "description": (
            "Run an in-depth multi-agent investigation on a complex question. "
            "Decomposes the question into independent research threads, dispatches "
            "sub-agents to gather data, evaluates completeness, and synthesizes a "
            "comprehensive answer. Use this for complex questions that need multiple "
            "angles of research (e.g., stock analysis, technology comparisons, "
            "strategic decisions)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The complex question to investigate thoroughly",
                },
            },
            "required": ["question"],
        },
    },
}


def execute(question: str) -> str:
    """Run multi-agent investigation."""
    return orchestrator.investigate(question)
