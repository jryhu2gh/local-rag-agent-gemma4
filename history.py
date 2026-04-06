"""Chat history persistence, indexing, and search.

Saves Q&A pairs to disk and indexes them for retrieval across sessions.
Reuses the existing BM25Index and VectorIndex classes with separate files.
"""

import json
from datetime import datetime
from pathlib import Path

from config import HISTORY_DIR, HISTORY_INDEX_DIR
from rag.bm25 import BM25Index
from rag.embedder import embed_one
from rag.index import VectorIndex

CONVERSATIONS_FILE = HISTORY_DIR / "conversations.jsonl"


def save_turn(user_message: str, agent_response: str):
    """Save a Q&A turn to disk and index it for search."""
    timestamp = datetime.now().isoformat(timespec="seconds")

    # 1. Append raw Q&A to JSONL file
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": timestamp,
        "user": user_message,
        "assistant": agent_response,
    }
    with open(CONVERSATIONS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 2. Index the Q&A pair for search
    chunk_text = f"Q: {user_message}\nA: {agent_response}"
    metadata = {
        "text": chunk_text,
        "timestamp": timestamp,
        "user_message": user_message,
        "source": "chat_history",
    }

    # Load or create history indexes
    HISTORY_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    bm25_path = HISTORY_INDEX_DIR / "bm25.json"
    if bm25_path.exists():
        bm25 = BM25Index.load(bm25_path)
    else:
        bm25 = BM25Index()

    vector = VectorIndex.load(HISTORY_INDEX_DIR)

    # Add to both indexes
    bm25.add(chunk_text, metadata=metadata)

    embedding = embed_one(chunk_text)
    vector.add(embedding, metadata=metadata)

    # Save
    bm25.save(bm25_path)
    vector.save(HISTORY_INDEX_DIR)


def search(query: str, top_k: int = 5) -> list[dict]:
    """Search chat history for past Q&A pairs relevant to the query."""
    bm25_path = HISTORY_INDEX_DIR / "bm25.json"
    if not bm25_path.exists():
        return []

    bm25 = BM25Index.load(bm25_path)
    vector = VectorIndex.load(HISTORY_INDEX_DIR)

    # Search both and merge via simple RRF
    bm25_results = bm25.search(query, top_k=top_k * 2)
    query_vec = embed_one(query)
    embed_results = vector.search(query_vec, top_k=top_k * 2)

    def result_key(r: dict) -> str:
        return r.get("timestamp", "") + r.get("user_message", "")

    rrf_k = 60
    scores: dict[str, float] = {}
    data: dict[str, dict] = {}

    for rank, r in enumerate(bm25_results):
        key = result_key(r)
        scores[key] = scores.get(key, 0) + 1.0 / (rrf_k + rank + 1)
        data[key] = r

    for rank, r in enumerate(embed_results):
        key = result_key(r)
        scores[key] = scores.get(key, 0) + 1.0 / (rrf_k + rank + 1)
        data[key] = r

    sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
    results = []
    for key in sorted_keys[:top_k]:
        entry = data[key]
        results.append({
            "text": entry.get("text", ""),
            "timestamp": entry.get("timestamp", ""),
            "score": round(scores[key], 4),
        })

    return results
