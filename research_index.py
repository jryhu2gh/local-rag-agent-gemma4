"""Session-scoped research index — in-memory only, discarded on exit.

Indexes raw research data (web pages, search results) gathered by sub-agents
during investigations. Searchable via reflect, enabling follow-up questions
to find data from earlier research without re-doing web searches.

Thread-safe: parallel sub-agents can call add() concurrently.
"""

import threading

from rag.bm25 import BM25Index
from rag.chunker import chunk_text
from rag.embedder import embed
from rag.hybrid import hybrid_search
from rag.index import VectorIndex

_lock = threading.Lock()
_bm25 = BM25Index()
_vector = VectorIndex()


def add(text: str, source: str, thread_id: str):
    """Chunk and index research content into the session index."""
    if not text or len(text.strip()) < 20:
        return
    chunks = chunk_text(text, source_file=source, doc_id=f"research:{thread_id}:{source}")
    if not chunks:
        return

    texts = [c["text"] for c in chunks]
    embeddings = embed(texts)

    with _lock:
        for chunk in chunks:
            _bm25.add(chunk["text"], metadata=chunk)
        _vector.add_batch(embeddings, chunks)


def search(query: str, top_k: int = 5) -> list[dict]:
    """Hybrid search over session research data."""
    if not _bm25.docs:
        return []
    return hybrid_search(query, _bm25, _vector, top_k=top_k)


def clear():
    """Reset the session index."""
    global _bm25, _vector
    with _lock:
        _bm25 = BM25Index()
        _vector = VectorIndex()


def size() -> int:
    """Number of chunks in the session index."""
    return len(_bm25.docs)
