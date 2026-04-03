"""Numpy-based vector store with cosine similarity search."""

import json
from pathlib import Path

import numpy as np


class VectorIndex:
    """In-memory vector index backed by numpy arrays."""

    def __init__(self):
        self.embeddings: np.ndarray | None = None  # (N, dim) float32
        self.metadata: list[dict] = []

    def add(self, embedding: list[float], metadata: dict):
        vec = np.array(embedding, dtype=np.float32).reshape(1, -1)
        if self.embeddings is None:
            self.embeddings = vec
        else:
            self.embeddings = np.vstack([self.embeddings, vec])
        self.metadata.append(metadata)

    def add_batch(self, embeddings: list[list[float]], metadatas: list[dict]):
        vecs = np.array(embeddings, dtype=np.float32)
        if self.embeddings is None:
            self.embeddings = vecs
        else:
            self.embeddings = np.vstack([self.embeddings, vecs])
        self.metadata.extend(metadatas)

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """Return top_k most similar chunks with scores."""
        if self.embeddings is None or len(self.metadata) == 0:
            return []

        query = np.array(query_embedding, dtype=np.float32)

        # Cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1)
        query_norm = np.linalg.norm(query)
        similarities = self.embeddings @ query / (norms * query_norm + 1e-10)

        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for i in top_indices:
            result = {**self.metadata[i], "score": float(similarities[i])}
            results.append(result)
        return results

    def save(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        if self.embeddings is not None:
            np.save(directory / "embeddings.npy", self.embeddings)
        (directory / "metadata.json").write_text(json.dumps(self.metadata))

    @classmethod
    def load(cls, directory: Path) -> "VectorIndex":
        idx = cls()
        emb_path = directory / "embeddings.npy"
        meta_path = directory / "metadata.json"
        if emb_path.exists():
            idx.embeddings = np.load(emb_path)
        if meta_path.exists():
            idx.metadata = json.loads(meta_path.read_text())
        return idx
