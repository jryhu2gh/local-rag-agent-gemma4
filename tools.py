"""Tool definitions and execution dispatch for the RAG agent."""

import json
from pathlib import Path

from config import DOCUMENTS_DIR, INDEX_DIR
from rag.bm25 import BM25Index
from rag.hybrid import hybrid_search
from rag.index import VectorIndex

# OpenAI-format tool definitions (passed to the API)
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search the document collection for passages relevant to a query. "
                "Uses hybrid search (keyword + semantic) to find the best matching chunks. "
                "Returns the top matching text chunks with scores and source info."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant document passages",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_document",
            "description": "Read the full content of a specific document by its ID (relative file path).",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "The document identifier (relative path) returned by search_documents",
                    }
                },
                "required": ["doc_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_text",
            "description": "Summarize a long piece of text into a concise form. Extracts the most important sentences.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text to summarize",
                    }
                },
                "required": ["text"],
            },
        },
    },
]


def _load_indexes() -> tuple[BM25Index, VectorIndex]:
    bm25 = BM25Index.load(INDEX_DIR / "bm25.json")
    vector = VectorIndex.load(INDEX_DIR)
    return bm25, vector


def _handle_search(query: str) -> str:
    bm25, vector = _load_indexes()
    results = hybrid_search(query, bm25, vector)
    if not results:
        return json.dumps({"results": [], "message": "No matching documents found."})

    formatted = []
    for r in results:
        formatted.append({
            "doc_id": r["doc_id"],
            "source_file": r["source_file"],
            "chunk_index": r["chunk_index"],
            "score": round(r["score"], 4),
            "text": r["text"][:500],  # truncate long chunks for the LLM
        })
    return json.dumps({"results": formatted}, indent=2)


def _handle_read(doc_id: str) -> str:
    file_path = DOCUMENTS_DIR / doc_id
    if not file_path.exists():
        return json.dumps({"error": f"Document not found: {doc_id}"})

    if file_path.suffix == ".pdf":
        import fitz
        doc = fitz.open(file_path)
        text = "\n\n".join(page.get_text() for page in doc)
        doc.close()
    else:
        text = file_path.read_text(encoding="utf-8", errors="replace")

    # Cap at 4000 chars to avoid flooding the context
    if len(text) > 4000:
        text = text[:4000] + f"\n\n[... truncated, {len(text)} chars total]"

    return json.dumps({"doc_id": doc_id, "content": text})


def _handle_summarize(text: str) -> str:
    """Extractive summarization: return first N sentences."""
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    max_sentences = min(10, len(sentences))
    summary = " ".join(sentences[:max_sentences])
    return json.dumps({"summary": summary, "sentence_count": max_sentences})


def execute_tool(name: str, arguments: dict) -> str:
    """Dispatch a tool call to the appropriate handler."""
    try:
        if name == "search_documents":
            return _handle_search(arguments["query"])
        elif name == "read_document":
            return _handle_read(arguments["doc_id"])
        elif name == "summarize_text":
            return _handle_summarize(arguments["text"])
        else:
            return json.dumps({"error": f"Unknown tool: {name}"})
    except Exception as e:
        return json.dumps({"error": f"Tool '{name}' failed: {str(e)}"})
