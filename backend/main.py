"""FastAPI backend.   Run:  uvicorn backend.main:app --port 8000

POST   /ingest         upload files (images, docs, tables) -> index them
POST   /chat           ask a question (optional images) -> streamed NDJSON
GET    /sources        list indexed files
DELETE /sources/{name} remove one file
DELETE /sources        clear the knowledge base
GET    /health         status, models, allowed file types
Interactive docs:      http://localhost:8000/docs
"""
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import TypeAdapter, ValidationError

from backend import config, llm, rag
from backend.schemas import (
    HealthResponse, IngestResponse, IngestResult, Message, SourcesResponse,
)
from backend.vectorstore import VectorStore

log = logging.getLogger("backend")
HISTORY = TypeAdapter(list[Message])


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.store = VectorStore()      # loads the embedding model once at startup
    yield


app = FastAPI(title="Multimodal RAG", version="1.0.0", lifespan=lifespan)


def get_store(request: Request) -> VectorStore:
    return request.app.state.store


def _event(kind: str, data) -> str:
    """One line of newline-delimited JSON: {"type": ..., "data": ...}"""
    return json.dumps({"type": kind, "data": data}, ensure_ascii=False) + "\n"


#  health
@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        groq_key_set=llm.has_key(),
        models={"vision": config.VISION_MODEL, "answer": config.TEXT_MODEL, "rewrite": config.FAST_MODEL},
        embed_model=config.EMBED_MODEL,
        allowed_types=config.ALL_TYPES,
    )


#  ingest
@app.post("/ingest", response_model=IngestResponse)
def ingest(files: list[UploadFile] = File(...), store: VectorStore = Depends(get_store)):
    results = []
    for f in files:
        name = Path(f.filename or "upload").name
        try:
            ext = Path(name).suffix.lower().lstrip(".")
            if ext not in config.ALL_TYPES:
                raise ValueError(f"Unsupported file type .{ext}")
            data = f.file.read()
            if len(data) > config.MAX_UPLOAD_MB * 1024 * 1024:
                raise ValueError(f"File is larger than {config.MAX_UPLOAD_MB} MB")
            results.append(IngestResult(name=name, chunks=rag.ingest_file(store, name, data)))
        except Exception as e:            # one bad file must not fail the whole batch
            log.warning("ingest failed for %s: %s", name, e)
            results.append(IngestResult(name=name, chunks=0, error=str(e)))
    return IngestResponse(results=results)


#  chat
@app.post("/chat")
# JSON list of {role, content}
# images sent with THIS question
def chat(message: str = Form(...), history: str = Form("[]"), top_k: int = Form(config.DEFAULT_TOP_K), images: list[UploadFile] = File(default=[]), store: VectorStore = Depends(get_store),
):
    if not llm.has_key():
        raise HTTPException(400, "GROQ_API_KEY is not set on the backend.")
    try:
        hist = [m.model_dump() for m in HISTORY.validate_json(history)]
    except ValidationError:
        raise HTTPException(422, "history must be a JSON list of {role, content} objects.")

    image_bytes = [f.file.read() for f in images]
    try:
        hits, stream = rag.ask(store, message, hist, image_bytes, max(1, min(top_k, config.MAX_TOP_K)))
    except Exception as e:
        log.exception("retrieval failed")
        raise HTTPException(502, f"Retrieval failed: {e}")

    def events():
        yield _event("sources", hits)
        try:
            for token in stream:
                yield _event("token", token)
        except Exception as e:            # headers are already sent, so report in-band
            log.exception("generation failed")
            yield _event("error", str(e))

    return StreamingResponse(events(), media_type="application/x-ndjson")


#  sources
@app.get("/sources", response_model=SourcesResponse)
def list_sources(store: VectorStore = Depends(get_store)):
    return SourcesResponse(sources=store.sources())


@app.delete("/sources/{name:path}")
def delete_source(name: str, store: VectorStore = Depends(get_store)):
    store.delete_source(name)
    return {"deleted": name}


@app.delete("/sources")
def clear_sources(store: VectorStore = Depends(get_store)):
    store.clear()
    return {"cleared": True}
