"""Turn any uploaded file into text elements, then into overlapping chunks."""
from __future__ import annotations

from dataclasses import dataclass
from docx import Document                           
from io import BytesIO
from pathlib import Path
from typing import Iterable, Iterator
import pdfplumber  
from backend import config, llm
import pandas as pd  


@dataclass
class Element:
    text: str
    kind: str          # "text" | "table" | "image"
    page: int = 0      # 0 = not applicable


# helpers
def rows_to_markdown(rows: list[list]) -> str:
    """List-of-rows (from pdfplumber / docx) -> markdown table."""
    rows = [[("" if c is None else str(c).replace("\n", " ").strip()) for c in r] for r in rows if r]
    if len(rows) < 2:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def dataframe_elements(df, label: str) -> list[Element]:
    df = df.dropna(how="all").dropna(axis=1, how="all").head(2000)
    if df.empty:
        return []
    summary = (
        f"Table '{label}': {len(df)} rows, {len(df.columns)} columns. "
        f"Columns: {', '.join(map(str, df.columns))}."
    )
    return [Element(summary, "text"), Element(df.fillna("").to_markdown(index=False), "table")]


# per-format loaders
def iter_pdf(data: bytes) -> Iterator[Element]:
    """Yield one page's elements at a time so memory stays flat."""

    vision_used = 0
    with pdfplumber.open(BytesIO(data)) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            try:
                text = (page.extract_text() or "").strip()
                for tbl in page.extract_tables():
                    md = rows_to_markdown(tbl)
                    if md:
                        yield Element(md, "table", i)
                if len(text) >= 30:
                    yield Element(text, "text", i)
                elif vision_used < config.MAX_VISION_PDF_PAGES:   # scanned page -> vision model
                    img = page.to_image(resolution=110)
                    try:
                        buf = BytesIO()
                        img.original.save(buf, "PNG")
                        png = buf.getvalue()
                    finally:
                        img.original.close()
                    del buf, img
                    yield Element(llm.describe_image(png), "image", i)
                    vision_used += 1
            finally:
                page.flush_cache()                      # drop this page's parsed objects


def load_docx(data: bytes) -> list[Element]:

    doc = Document(BytesIO(data))
    out = []
    body = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    if body:
        out.append(Element(body, "text"))
    for t in doc.tables:
        md = rows_to_markdown([[c.text for c in r.cells] for r in t.rows])
        if md:
            out.append(Element(md, "table"))
    return out


def load_csv(data: bytes, name: str) -> list[Element]:

    return dataframe_elements(pd.read_csv(BytesIO(data), nrows=5000), name)


def load_xlsx(data: bytes, name: str) -> list[Element]:

    out = []
    for sheet, df in pd.read_excel(BytesIO(data), sheet_name=None, nrows=5000).items():
        out += dataframe_elements(df, f"{name} / {sheet}")
    return out


def load_text(data: bytes) -> list[Element]:
    return [Element(data.decode("utf-8", errors="ignore"), "text")]


def load_image(data: bytes) -> list[Element]:
    return [Element(llm.describe_image(data), "image")]


def iter_elements(name: str, data: bytes) -> Iterator[Element]:
    ext = Path(name).suffix.lower().lstrip(".")
    if ext == "pdf":
        yield from iter_pdf(data)
    elif ext == "docx":
        yield from load_docx(data)
    elif ext == "csv":
        yield from load_csv(data, name)
    elif ext == "xlsx":
        yield from load_xlsx(data, name)
    elif ext in ("txt", "md"):
        yield from load_text(data)
    elif ext in config.IMAGE_TYPES:
        yield from load_image(data)
    else:
        raise ValueError(f"Unsupported file type: .{ext}")


def load_file(name: str, data: bytes) -> list[Element]:   # kept for compatibility
    return list(iter_elements(name, data))


# chunking
def split_text(text: str, size: int, overlap: int) -> list[str]:
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = max(text.rfind("\n\n", start, end), text.rfind(". ", start, end))
            if cut > start + size // 2:
                end = cut + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


def split_table(md: str, rows_per_chunk: int) -> list[str]:
    """Keep the header on every chunk so each piece is self-explanatory."""
    lines = md.splitlines()
    if len(lines) <= 2 + rows_per_chunk:
        return [md]
    header, body = lines[:2], lines[2:]
    return ["\n".join(header + body[i:i + rows_per_chunk]) for i in range(0, len(body), rows_per_chunk)]


def iter_chunks(elements: Iterable[Element], source: str) -> Iterator[dict]:
    for el in elements:
        if el.kind == "table":
            pieces = split_table(el.text, config.TABLE_ROWS_PER_CHUNK)
        else:
            pieces = split_text(el.text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        for p in pieces:
            yield {"text": p, "meta": {"source": source, "page": el.page, "kind": el.kind}}


def chunk_elements(elements: Iterable[Element], source: str) -> list[dict]:  # compatibility
    return list(iter_chunks(elements, source))