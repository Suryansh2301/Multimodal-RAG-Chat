
import re

import chromadb

from backend import config
from backend.embeddings import embed_query, embed_texts

try:
    from backend.memguard import check_memory
except ImportError:                         
    def check_memory() -> None:
        pass


class VectorStore:
    def __init__(self):
        config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        self._db = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", config.EMBED_MODEL).strip("_")[:40]
        self._name = f"kb_{slug}_{config.EMBED_DIM}"
        self.last_source = None
        self.col = self._db.get_or_create_collection(
            self._name, metadata={"hnsw:space": "cosine"}
        )

    def add(self, source: str, chunks: list[dict]) -> int:
        return self.add_stream(source, iter(chunks))

    def add_stream(self, source: str, chunks) -> int:
        self.delete_source(source)           # re-uploading a file replaces it
        total, batch = 0, []
        for c in chunks:
            batch.append(c)
            if len(batch) >= config.ADD_BATCH:
                total += self._flush(source, batch, total)
                batch = []
        if batch:
            total += self._flush(source, batch, total)
        if total:
            self.last_source = source
        return total

    def _flush(self, source: str, batch: list[dict], offset: int) -> int:
        check_memory()
        docs = [c["text"] for c in batch]
        self.col.add(
            ids=[f"{source}::{offset + i}" for i in range(len(batch))],
            documents=docs,
            embeddings=embed_texts(docs),
            metadatas=[c["meta"] for c in batch],
        )
        return len(batch)

    def search(self, query: str, k: int) -> list[dict]:
        n = self.col.count()
        if n == 0:
            return []
        res = self.col.query(query_embeddings=[embed_query(query)], n_results=min(k, n))
        return [
            {"text": d, "meta": m, "distance": dist}
            for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
        ]

    def sample(self, source: str, n: int) -> list[dict]:
        """Evenly spaced chunks across a file, in page order (for summaries)."""
        res = self.col.get(where={"source": source}, include=["documents", "metadatas"], limit=500)
        items = sorted(zip(res["documents"], res["metadatas"]), key=lambda x: x[1].get("page", 0))
        if not items:
            return []
        step = max(1, len(items) // n)
        return [{"text": d, "meta": m, "distance": 0.0} for d, m in items[::step][:n]]

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