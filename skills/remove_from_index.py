"""Remove from index skill — remove a document's indexed data without deleting the file."""

import json
from pathlib import Path

from config import INDEX_DIR
from rag.bm25 import BM25Index
from rag.index import VectorIndex

DEFINITION = {
    "type": "function",
    "function": {
        "name": "remove_from_index",
        "description": (
            "Remove a document from the persistent search index so it no longer "
            "appears in reflect results. The raw file on disk is NOT deleted. "
            "Accepts either a doc_id (as stored in the index) or a file path."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "doc_id": {
                    "type": "string",
                    "description": (
                        "The document ID or file path to remove. Can be a relative name "
                        "(e.g. 'sample-python.md'), an absolute path, or a path with ~ "
                        "which will be expanded."
                    ),
                },
            },
            "required": ["doc_id"],
        },
    },
}


def _collect_matching_doc_ids(doc_id: str, all_doc_ids: set[str]) -> set[str]:
    """Find all doc_ids in the index that match the user-provided identifier."""
    matches = set()

    # Direct match
    if doc_id in all_doc_ids:
        matches.add(doc_id)

    # Try resolving as a path (expand ~ and resolve)
    try:
        resolved = str(Path(doc_id).expanduser().resolve())
        if resolved in all_doc_ids:
            matches.add(resolved)
    except Exception:
        pass

    # Match by filename (e.g. "test-deep-sea-research.md" matches any doc_id ending with it)
    for existing in all_doc_ids:
        if existing.endswith("/" + doc_id) or existing.endswith("\\" + doc_id):
            matches.add(existing)
        if Path(existing).name == doc_id:
            matches.add(existing)

    return matches


def execute(doc_id: str) -> str:
    """Remove a document from the BM25 and vector indexes."""
    bm25_path = INDEX_DIR / "bm25.json"
    if not bm25_path.exists():
        return json.dumps({"error": "No index found. Nothing to remove."})

    bm25 = BM25Index.load(bm25_path)
    vector = VectorIndex.load(INDEX_DIR)

    # Gather all known doc_ids from both indexes
    all_doc_ids = set()
    all_doc_ids.update(d.get("doc_id", "") for d in bm25.docs)
    all_doc_ids.update(m.get("doc_id", "") for m in vector.metadata)
    all_doc_ids.discard("")

    matches = _collect_matching_doc_ids(doc_id, all_doc_ids)

    if not matches:
        return json.dumps({
            "error": f"No indexed document matching '{doc_id}'",
            "indexed_doc_ids": sorted(all_doc_ids),
        }, indent=2)

    total_bm25 = 0
    total_vector = 0
    for mid in matches:
        total_bm25 += bm25.remove_by_doc_id(mid)
        total_vector += vector.remove_by_doc_id(mid)

    bm25.save(bm25_path)
    vector.save(INDEX_DIR)

    print(f"  [remove_from_index] Removed {total_bm25} BM25 + {total_vector} vector chunks for {matches}")

    return json.dumps({
        "removed_doc_ids": sorted(matches),
        "bm25_chunks_removed": total_bm25,
        "vector_chunks_removed": total_vector,
        "remaining_index_size": len(bm25.docs),
    }, indent=2)
