""" ingest a file, then retrieve + answer."""
from typing import Iterator

from backend import ingest, llm
from backend.vectorstore import VectorStore


def ingest_file(store: VectorStore, name: str, data: bytes) -> int:
    elements = ingest.load_file(name, data)
    chunks = ingest.chunk_elements(elements, source=name)
    return store.add(name, chunks)


def build_context(hits: list[dict]) -> str:
    parts = []
    for i, h in enumerate(hits, start=1):
        m = h["meta"]
        where = m["source"] + (f", p.{m['page']}" if m["page"] else "") + f", {m['kind']}"
        parts.append(f"[{i}] ({where})\n{h['text']}")
    return "\n\n".join(parts)


def ask(store: VectorStore, question: str, history: list[dict], images: list[bytes],top_k: int) -> tuple[list[dict], Iterator[str]]:
    """Returns (retrieved hits, streaming answer generator)."""
    query = llm.rewrite_query(question, history)
    hits = store.search(query, top_k)
    stream = llm.stream_answer(question, build_context(hits), history, images)
    return hits, stream
