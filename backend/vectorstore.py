"""Thin ChromaDB wrapper. The collection name includes the embedding model,
so switching EMBED_MODEL never causes a dimension-mismatch error."""
import re

import chromadb
from fastembed import TextEmbedding

from backend import config


class VectorStore:
    def __init__(self):
        config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self._db = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        self._embedder = TextEmbedding(model_name=config.EMBED_MODEL)   # ONNX, no torch
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", config.EMBED_MODEL).strip("_")[:40]
        self._name = f"kb_{slug}"
        self.col = self._db.get_or_create_collection(
            self._name, metadata={"hnsw:space": "cosine"}
        )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._embedder.embed(texts)]

    def add(self, source: str, chunks: list[dict]) -> int:
        return self.add_stream(source, iter(chunks))

    def add_stream(self, source: str, chunks) -> int:
        """Consume chunks lazily, embedding and storing ADD_BATCH at a time."""
        self.delete_source(source)               # re-uploading a file replaces it
        total, batch = 0, []
        for c in chunks:
            batch.append(c)
            if len(batch) >= config.ADD_BATCH:
                total += self._flush(source, batch, total)
                batch = []
        if batch:
            total += self._flush(source, batch, total)
        return total

    def _flush(self, source: str, batch: list[dict], offset: int) -> int:
        docs = [c["text"] for c in batch]
        self.col.add(
            ids=[f"{source}::{offset + i}" for i in range(len(batch))],
            documents=docs,
            embeddings=self._embed(docs),
            metadatas=[c["meta"] for c in batch],
        )
        return len(batch)
    
    def search(self, query: str, k: int) -> list[dict]:
        n = self.col.count()
        if n == 0:
            return []
        qv = next(iter(self._embedder.query_embed([query]))).tolist()
        res = self.col.query(query_embeddings=[qv], n_results=min(k, n))
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
            self._name, metadata={"hnsw:space": "cosine"}
        )