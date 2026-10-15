"""Thin ChromaDB wrapper. The collection name includes the embedding model,
so switching EMBED_MODEL never causes a dimension-mismatch error."""
import re

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from backend import config


class VectorStore:
    def __init__(self):
        config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self._db = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        self._ef = SentenceTransformerEmbeddingFunction(model_name=config.EMBED_MODEL)
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", config.EMBED_MODEL).strip("_")[:40]
        self._name = f"kb_{slug}"
        self.col = self._db.get_or_create_collection(
            self._name, embedding_function=self._ef, metadata={"hnsw:space": "cosine"}
        )

    def add(self, source: str, chunks: list[dict]) -> int:
        self.delete_source(source)               # re-uploading a file replaces it
        if not chunks:
            return 0
        self.col.add(
            ids=[f"{source}::{i}" for i in range(len(chunks))],
            documents=[c["text"] for c in chunks],
            metadatas=[c["meta"] for c in chunks],
        )
        return len(chunks)

    def search(self, query: str, k: int) -> list[dict]:
        n = self.col.count()
        if n == 0:
            return []
        res = self.col.query(query_texts=[query], n_results=min(k, n))
        return [
            {"text": d, "meta": m, "distance": dist}
            for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]

    def sources(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for m in self.col.get(include=["metadatas"])["metadatas"]:
            counts[m["source"]] = counts.get(m["source"], 0) + 1
        return counts

    def delete_source(self, source: str) -> None:
        self.col.delete(where={"source": source})

    def clear(self) -> None:
        self._db.delete_collection(self._name)
        self.col = self._db.get_or_create_collection(
            self._name, embedding_function=self._ef, metadata={"hnsw:space": "cosine"}
        )
