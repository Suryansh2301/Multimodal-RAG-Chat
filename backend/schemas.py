"""Request/response models (they also drive the auto-generated /docs page)."""
from typing import Literal

from pydantic import BaseModel


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class HealthResponse(BaseModel):
    status: str
    groq_key_set: bool
    models: dict[str, str]
    embed_model: str
    allowed_types: list[str]


class IngestResult(BaseModel):
    name: str
    chunks: int
    error: str | None = None


class IngestResponse(BaseModel):
    results: list[IngestResult]


class SourcesResponse(BaseModel):
    sources: dict[str, int]   # file name -> number of chunks
