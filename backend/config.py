"""Central settings. Everything can be overridden from .env."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "data" / "chroma"

# --- Groq models 
VISION_MODEL = os.getenv("VISION_MODEL", "qwen/qwen3.6-27b")   # images, scans, charts
TEXT_MODEL = os.getenv("TEXT_MODEL", "openai/gpt-oss-120b")    # final answers
FAST_MODEL = os.getenv("FAST_MODEL", "openai/gpt-oss-20b")     # query rewriting

# --- Embeddings 
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# --- Chunking / retrieval
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TABLE_ROWS_PER_CHUNK = 25
DEFAULT_TOP_K = 5
MAX_VISION_PDF_PAGES = 8      # scanned PDF pages sent to the vision model
HISTORY_TURNS = 6             # past messages sent to the LLM

# --- Supported uploads 
IMAGE_TYPES = ["png", "jpg", "jpeg", "webp"]
DOC_TYPES = ["pdf", "docx", "txt", "md"]
TABLE_TYPES = ["csv", "xlsx"]
ALL_TYPES = IMAGE_TYPES + DOC_TYPES + TABLE_TYPES

# --- API ----
MAX_UPLOAD_MB = 25
MAX_TOP_K = 20
