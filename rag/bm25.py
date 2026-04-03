"""BM25 keyword search index. Pure Python, no external dependencies."""

import json
import math
import re
from collections import Counter
from pathlib import Path


def _tokenize(text: str) -> list[str]:
    """Lowercase and split into word tokens."""
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    """Okapi BM25 ranking over text chunks."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: list[dict] = []          # chunk metadata
        self.doc_tokens: list[list[str]] = []
        self.doc_freqs: dict[str, int] = {}  # term -> number of docs containing it
        self.avg_dl: float = 0.0

    def add(self, text: str, metadata: dict):
        tokens = _tokenize(text)
        self.docs.append(metadata)
        self.doc_tokens.append(tokens)

        # Update document frequencies
        for term in set(tokens):
            self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1

        # Recompute average doc length
        total = sum(len(t) for t in self.doc_tokens)
        self.avg_dl = total / len(self.doc_tokens)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search for query, return top_k results with scores."""
        query_tokens = _tokenize(query)
        if not query_tokens or not self.docs:
            return []

        n = len(self.docs)
        scores = []

        for i, doc_tokens in enumerate(self.doc_tokens):
            tf_map = Counter(doc_tokens)
            dl = len(doc_tokens)
            score = 0.0

            for term in query_tokens:
                if term not in self.doc_freqs:
                    continue
                tf = tf_map.get(term, 0)
                if tf == 0:
                    continue
                df = self.doc_freqs[term]
                idf = math.log((n - df + 0.5) / (df + 0.5) + 1)
                tf_norm = (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * dl / self.avg_dl))
                score += idf * tf_norm

            if score > 0:
                result = {**self.docs[i], "score": score}
                scores.append(result)

        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores[:top_k]

    def save(self, path: Path):
        data = {
            "k1": self.k1,
            "b": self.b,
            "docs": self.docs,
            "doc_tokens": self.doc_tokens,
            "doc_freqs": self.doc_freqs,
            "avg_dl": self.avg_dl,
        }
        path.write_text(json.dumps(data))

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        data = json.loads(path.read_text())
        idx = cls(k1=data["k1"], b=data["b"])
        idx.docs = data["docs"]
        idx.doc_tokens = data["doc_tokens"]
        idx.doc_freqs = data["doc_freqs"]
        idx.avg_dl = data["avg_dl"]
        return idx
