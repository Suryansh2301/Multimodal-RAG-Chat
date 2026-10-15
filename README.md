# Multimodal RAG Chat — FastAPI backend + Streamlit frontend

One chat box. Type a question, or use the attach button on the left of the box to add
**images, documents (PDF/DOCX/TXT/MD) and tables (CSV/XLSX)**. Files are read, indexed and used
to answer your questions with cited sources. Models run on Groq.

## Architecture
```
Streamlit (frontend/app.py)  ── HTTP ──►  FastAPI (backend/main.py)
   chat UI + attach button                 /ingest  /chat  /sources  /health
                                                │
                                   ┌────────────┼─────────────┐
                               ingest.py      rag.py        llm.py ──► Groq
                                   └──► vectorstore.py (ChromaDB + local embeddings)
```
The frontend only makes HTTP calls. The Groq key, models, embeddings and database all live in the backend.

## Structure
```
multimodal-rag-chatbot/
├── backend/
│   ├── main.py            # FastAPI app and routes
│   ├── schemas.py         # request/response models
│   ├── config.py          # model names, chunk sizes, allowed file types
│   ├── llm.py             # Groq calls: vision OCR, query rewrite, streamed answer
│   ├── ingest.py          # PDF/DOCX/CSV/XLSX/image -> text -> chunks
│   ├── vectorstore.py     # ChromaDB + local embeddings
│   ├── rag.py             # ingest_file() and ask()
│   └── requirements.txt
├── frontend/
│   ├── app.py             # Streamlit chat UI
│   ├── api_client.py      # HTTP client for the backend
│   └── requirements.txt
├── data/                  # ChromaDB storage (git-ignored)
├── .env.example           # copy to .env and add GROQ_API_KEY
└── .streamlit/config.toml # theme
```

## Run (two terminals, from the project root)
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r backend/requirements.txt -r frontend/requirements.txt
cp .env.example .env                                   # paste your free key from console.groq.com/keys

# terminal 1: backend
uvicorn backend.main:app --port 8000

# terminal 2: frontend
streamlit run frontend/app.py
```
Open http://localhost:8501 for the chat and http://localhost:8000/docs for the interactive API docs.
If the backend runs elsewhere, set `API_URL=http://host:port` before starting Streamlit.

## API
| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | status, model names, allowed file types |
| POST | `/ingest` | multipart `files` -> `{results: [{name, chunks, error}]}` |
| POST | `/chat` | form: `message`, `history` (JSON), `top_k`, optional `images` -> streamed NDJSON |
| GET | `/sources` | indexed files and chunk counts |
| DELETE | `/sources/{name}` | remove one file |
| DELETE | `/sources` | clear the knowledge base |

`/chat` streams one JSON object per line: `{"type":"sources","data":[...]}` first, then
`{"type":"token","data":"..."}` repeatedly, or `{"type":"error","data":"..."}` if generation fails.

```bash
curl -F "files=@report.pdf" http://localhost:8000/ingest
curl -N -F "message=What changed in Q3?" -F 'history=[]' http://localhost:8000/chat
```

## Models
| Step | Model |
|---|---|
| Images and scanned PDF pages -> markdown (OCR, tables, charts) | `qwen/qwen3.6-27b` |
| Follow-up question rewriting | `openai/gpt-oss-20b` |
| Final cited answer (streamed) | `openai/gpt-oss-120b` |
| Question with an attached image (direct visual Q&A) | `qwen/qwen3.6-27b` |
| Embeddings (local) | `BAAI/bge-small-en-v1.5` |

## Notes
- Qwen vision models on Groq are **Preview**; swap with `VISION_MODEL` in `.env`.
- `EMBED_MODEL=BAAI/bge-m3` gives better retrieval but is heavier. The collection name includes the
  embedding model, so switching never causes a dimension mismatch.
- The backend loads the embedding model at startup, so the first launch takes a moment.
- Images attached together with a question are sent to `/ingest` (to be indexed) and to `/chat` (for direct Q&A).
- Upgrade ideas: BM25 hybrid search, a cross-encoder reranker, per-user collections, an API token between frontend and backend.
