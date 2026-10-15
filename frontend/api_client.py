"""Tiny HTTP client for the FastAPI backend. The UI never touches models or the DB directly."""
import json
import os
from typing import Iterator

import requests
from urllib.parse import quote

# if you are running the frontend in a separate container, set API_URL to the backend container name (e.g. http://backend:8000)
# API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
API_URL = os.getenv("API_URL", "https://multimodal-rag-chat.onrender.com").rstrip("/")


class APIError(Exception):
    pass


def _request(method: str, path: str, timeout=30, **kwargs) -> requests.Response:
    try:
        r = requests.request(method, f"{API_URL}{path}", timeout=timeout, **kwargs)
    except requests.RequestException as e:
        raise APIError(str(e)) from e
    if not r.ok:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise APIError(f"{r.status_code}: {detail}")
    return r


def health() -> dict:
    return _request("GET", "/health", timeout=5).json()


def list_sources() -> dict[str, int]:
    return _request("GET", "/sources").json()["sources"]


def delete_source(name: str) -> None:
    _request("DELETE", f"/sources/{quote(name, safe='')}")


def clear_sources() -> None:
    _request("DELETE", "/sources")


def ingest(files: list[tuple[str, bytes]]) -> list[dict]:
    """files = [(filename, bytes), ...] -> [{name, chunks, error}, ...]"""
    payload = [("files", (name, data)) for name, data in files]
    return _request("POST", "/ingest", timeout=600, files=payload).json()["results"]


class ChatStream:
    """Iterate .tokens() for the answer text; .sources is filled as soon as it arrives."""

    def __init__(self, response: requests.Response):
        self._r = response
        self._r.encoding = "utf-8"
        self.sources: list[dict] = []

    def tokens(self) -> Iterator[str]:
        try:
            for line in self._r.iter_lines(chunk_size=None, decode_unicode=True):
                if not line:
                    continue
                event = json.loads(line)
                if event["type"] == "sources":
                    self.sources = event["data"]
                elif event["type"] == "token":
                    yield event["data"]
                elif event["type"] == "error":
                    raise APIError(event["data"])
        finally:
            self._r.close()


def chat_stream( question: str, history: list[dict], images: list[tuple[str, bytes]], top_k: int) -> ChatStream:
    data = {"message": question, "history": json.dumps(history), "top_k": top_k}
    files = [("images", (name, blob)) for name, blob in images]
    r = _request("POST", "/chat", timeout=(5, 300), stream=True, data=data, files=files or None)
    return ChatStream(r)
