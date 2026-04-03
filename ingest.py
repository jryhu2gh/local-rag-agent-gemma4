#!/usr/bin/python3
"""Ingest documents into both BM25 and vector indexes.

Usage:
    python3 ingest.py                    # ingest from default documents/ dir
    python3 ingest.py --dir /path/to/docs
"""

import argparse
import sys
from pathlib import Path

# Ensure the agent directory is on the Python path
sys.path.insert(0, str(Path(__file__).parent))

from config import DOCUMENTS_DIR, INDEX_DIR
from rag.bm25 import BM25Index
from rag.chunker import chunk_text
from rag.embedder import embed
from rag.index import VectorIndex

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


def extract_text(file_path: Path) -> str:
    """Extract text from a file based on its extension."""
    if file_path.suffix == ".pdf":
        import fitz  # pymupdf
        doc = fitz.open(file_path)
        text = "\n\n".join(page.get_text() for page in doc)
        doc.close()
        return text
    else:
        return file_path.read_text(encoding="utf-8", errors="replace")


def ingest(docs_dir: Path):
    """Scan docs_dir, chunk, and build both indexes."""
    files = [
        f for f in sorted(docs_dir.rglob("*"))
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        print(f"No documents found in {docs_dir}")
        print(f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}")
        return

    print(f"Found {len(files)} document(s) in {docs_dir}")

    bm25 = BM25Index()
    vector = VectorIndex()
    all_chunks = []

    for file_path in files:
        rel_path = str(file_path.relative_to(docs_dir))
        doc_id = rel_path
        print(f"  Processing: {rel_path}")

        text = extract_text(file_path)
        if not text.strip():
            print(f"    Skipped (empty)")
            continue

        chunks = chunk_text(text, source_file=rel_path, doc_id=doc_id)
        print(f"    {len(chunks)} chunks")
        all_chunks.extend(chunks)

    if not all_chunks:
        print("No chunks to index.")
        return

    # Build BM25 index
    print(f"\nBuilding BM25 index ({len(all_chunks)} chunks)...")
    for chunk in all_chunks:
        bm25.add(chunk["text"], metadata=chunk)

    # Build vector index (batch embed for efficiency)
    print(f"Building vector index ({len(all_chunks)} chunks)...")
    batch_size = 32
    all_texts = [c["text"] for c in all_chunks]
    all_embeddings = []
    for i in range(0, len(all_texts), batch_size):
        batch = all_texts[i : i + batch_size]
        all_embeddings.extend(embed(batch))
        print(f"  Embedded {min(i + batch_size, len(all_texts))}/{len(all_texts)}")

    vector.add_batch(all_embeddings, all_chunks)

    # Save both indexes
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    bm25.save(INDEX_DIR / "bm25.json")
    vector.save(INDEX_DIR)
    print(f"\nDone! Indexes saved to {INDEX_DIR}")
    print(f"  BM25:      {INDEX_DIR / 'bm25.json'}")
    print(f"  Vectors:   {INDEX_DIR / 'embeddings.npy'}")
    print(f"  Metadata:  {INDEX_DIR / 'metadata.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents for RAG agent")
    parser.add_argument("--dir", type=Path, default=DOCUMENTS_DIR, help="Documents directory")
    args = parser.parse_args()
    ingest(args.dir)
