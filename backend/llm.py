"""All Groq calls live here: vision extraction, query rewriting, answer streaming."""
import base64
import os
import re
from functools import lru_cache
from io import BytesIO
from typing import Iterator

from groq import Groq
from PIL import Image, ImageOps

from backend import config

SYSTEM_PROMPT = (
    "You are a helpful assistant inside a chat app. The user may upload images, "
    "documents and tables; the relevant parts are provided below as numbered CONTEXT "
    "snippets. Use them when they help and cite them inline like [1], [2]. "
    "If the question depends on the uploaded files but the context does not contain "
    "the answer, say so plainly instead of guessing. If no context is provided, "
    "just answer like a normal assistant. Keep tables in markdown."
)

EXTRACT_PROMPT = (
    "Extract everything useful from this image so it can be searched later.\n"
    "1. Transcribe ALL visible text exactly.\n"
    "2. Reproduce every table as a GitHub-flavored markdown table (keep all rows/columns).\n"
    "3. For charts or diagrams, state the type, axes, series and the key values or trends.\n"
    "4. If it is a photo or screenshot, describe what it shows in 2-3 sentences.\n"
    "Output markdown only, no preamble."
)


@lru_cache(maxsize=4)
def _client_for(key: str) -> Groq:
    return Groq(api_key=key)


def has_key() -> bool:
    return bool(os.getenv("GROQ_API_KEY", "").strip())


def get_client() -> Groq:
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to .env and restart the backend.")
    return _client_for(key)


def _strip_think(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


def to_data_uri(image_bytes: bytes) -> str:
    """Downscale + re-encode as JPEG so requests stay small and always valid."""
    img = Image.open(BytesIO(image_bytes))
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, "white")
        bg.paste(img, mask=img.split()[-1])
        img = bg
    else:
        img = img.convert("RGB")
    img.thumbnail((1568, 1568))
    buf = BytesIO()
    img.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def describe_image(image_bytes: bytes) -> str:
    """Vision model -> markdown (OCR + tables + chart/photo description). Used at ingestion."""
    resp = get_client().chat.completions.create(
        model=config.VISION_MODEL,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": EXTRACT_PROMPT},
                {"type": "image_url", "image_url": {"url": to_data_uri(image_bytes)}},
            ],
        }],
        temperature=0.2,
        max_completion_tokens=2048,
        reasoning_effort="none",
    )
    return _strip_think(resp.choices[0].message.content)


def rewrite_query(question: str, history: list[dict]) -> str:
    """Make a follow-up like 'and the second one?' standalone for retrieval."""
    if not history:
        return question
    convo = "\n".join(f"{m['role']}: {m['content'][:400]}" for m in history[-4:])
    try:
        resp = get_client().chat.completions.create(
            model=config.FAST_MODEL,
            messages=[{
                "role": "user",
                "content": (
                    "Rewrite the last user question as one standalone search query, using the "
                    "conversation for missing context. Output the query only.\n\n"
                    f"Conversation:\n{convo}\n\nLast question: {question}"
                ),
            }],
            temperature=0.2,
            max_completion_tokens=512,   # gpt-oss reasoning tokens count toward this
            reasoning_effort="low",
        )
        return _strip_think(resp.choices[0].message.content) or question
    except Exception:
        return question


def stream_answer(question: str, context: str, history: list[dict], images: list[bytes] | None = None) -> Iterator[str]:
    """Stream the final answer. Uses the vision model only if images came with this message."""
    user_text = f"CONTEXT:\n{context}\n\nQUESTION: {question}" if context else question
    if images:
        content = [{"type": "text", "text": user_text}] + [
            {"type": "image_url", "image_url": {"url": to_data_uri(b)}} for b in images
        ]
        model, extra = config.VISION_MODEL, {"reasoning_effort": "none"}
    else:
        content = user_text
        model, extra = config.TEXT_MODEL, {"reasoning_effort": "low"}

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += [{"role": m["role"], "content": m["content"]} for m in history[-config.HISTORY_TURNS:]]
    messages.append({"role": "user", "content": content})

    stream = get_client().chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.4,
        max_completion_tokens=2048,
        stream=True,
        **extra,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content
