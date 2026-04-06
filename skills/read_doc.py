"""Read document skill — read the full content of a specific document."""

import toolkit

DEFINITION = {
    "type": "function",
    "function": {
        "name": "read_document",
        "description": (
            "Read the full content of a specific document by its ID. "
            "Use this after a research result to see the complete text of a document."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "doc_id": {
                    "type": "string",
                    "description": "The document identifier (relative path) from a prior research result",
                }
            },
            "required": ["doc_id"],
        },
    },
}


def execute(doc_id: str) -> str:
    """Read a document by ID."""
    return toolkit.read_document(doc_id)
