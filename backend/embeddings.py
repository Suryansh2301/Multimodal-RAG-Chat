"""Gemini embeddings with batching, a token-per-minute throttle and retries."""
import os
from google.genai import types
import threading
import time

from backend import config

_client = None
_lock = threading.Lock()
_window: list[tuple[float, int]] = []      # (timestamp, tokens) in the last 60s


def _get_client():
    global _client
    if _client is None:
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set on the backend.")
        from google import genai
        _client = genai.Client(api_key=key)
    return _client


def _throttle(tokens: int) -> None:
    with _lock:
        now = time.time()
        while _window and now - _window[0][0] > 60:
            _window.pop(0)
        if _window and sum(t for _, t in _window) + tokens > config.EMBED_TPM_BUDGET:
            time.sleep(max(60 - (now - _window[0][0]) + 0.5, 0))
        _window.append((time.time(), tokens))


def _call(contents: list[str]) -> list[list[float]]:
    for attempt in range(6):
        try:
            r = _get_client().models.embed_content(
                model=config.EMBED_MODEL,
                contents=contents,
                config=types.EmbedContentConfig(output_dimensionality=config.EMBED_DIM),
            )
            return [list(e.values) for e in r.embeddings]
        except Exception as e:
            msg = str(e)
            transient = any(s in msg for s in ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE"))
            if not transient or attempt == 5:
                raise RuntimeError(f"Gemini embedding failed: {msg[:300]}") from e
            time.sleep(min(2 ** attempt * 2, 30))


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    _throttle(sum(len(t) for t in texts) // 4 + len(texts))
    vecs = _call(texts)
    if len(vecs) != len(texts):            
        vecs = []
        for t in texts:
            _throttle(len(t) // 4 + 1)
            vecs.append(_call([t])[0])
    return vecs


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]