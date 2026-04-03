"""Hybrid search: merge BM25 and embedding results via Reciprocal Rank Fusion."""

from config import RRF_K, TOP_K
from rag.bm25 import BM25Index
from rag.index import VectorIndex
from rag.embedder import embed_one


def _rrf_score(rank: int, k: int = RRF_K) -> float:
    """Reciprocal Rank Fusion score for a given rank (0-based)."""
    return 1.0 / (k + rank + 1)


def hybrid_search(
    query: str,
    bm25_index: BM25Index,
    vector_index: VectorIndex,
    top_k: int = TOP_K,
) -> list[dict]:
    """Search using both BM25 and embeddings, merge with RRF.

    Returns top_k results sorted by combined RRF score.
    Each result has: text, source_file, doc_id, chunk_index, score,
    bm25_rank, embed_rank.
    """
    # Get results from both methods (fetch more than top_k for better fusion)
    fetch_k = top_k * 3
    bm25_results = bm25_index.search(query, top_k=fetch_k)
    query_vec = embed_one(query)
    embed_results = vector_index.search(query_vec, top_k=fetch_k)

    # Build unique key for each chunk
    def chunk_key(r: dict) -> str:
        return f"{r['doc_id']}:{r['chunk_index']}"

    # Compute RRF scores
    rrf_scores: dict[str, float] = {}
    chunk_data: dict[str, dict] = {}
    bm25_ranks: dict[str, int] = {}
    embed_ranks: dict[str, int] = {}

    for rank, r in enumerate(bm25_results):
        key = chunk_key(r)
        rrf_scores[key] = rrf_scores.get(key, 0) + _rrf_score(rank)
        chunk_data[key] = {k: v for k, v in r.items() if k != "score"}
        bm25_ranks[key] = rank

    for rank, r in enumerate(embed_results):
        key = chunk_key(r)
        rrf_scores[key] = rrf_scores.get(key, 0) + _rrf_score(rank)
        chunk_data[key] = {k: v for k, v in r.items() if k != "score"}
        embed_ranks[key] = rank

    # Sort by RRF score and return top_k
    sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)

    results = []
    for key in sorted_keys[:top_k]:
        result = {
            **chunk_data[key],
            "score": rrf_scores[key],
            "bm25_rank": bm25_ranks.get(key),
            "embed_rank": embed_ranks.get(key),
        }
        results.append(result)

    return results
