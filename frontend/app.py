
import streamlit as st

import api_client as api

st.set_page_config(page_title="QueryMind-ai", page_icon="💬", layout="centered")

st.markdown(
    """
    <style>
        .block-container {padding-top: 2rem; max-width: 820px;}
        .chip {display:inline-block; padding:2px 10px; margin:2px 6px 2px 0; border-radius:999px;
            background:#f0eee6; border:1px solid #e0ddd2; font-size:0.8rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

#  backend check
try:
    info = api.health()
except api.APIError as e:
    st.error(
        f"Can't reach the backend at `{api.API_URL}`.\n\n"
        "Start it in another terminal:\n\n`uvicorn backend.main:app --port 8000`"
    )
    st.caption(str(e))
    st.stop()

IMAGE_TYPES = {"png", "jpg", "jpeg", "webp"}
if "messages" not in st.session_state:
    st.session_state.messages = []

#  sidebar
with st.sidebar:
    st.header("Settings")
    if not info["groq_key_set"]:
        st.warning("The backend has no GROQ_API_KEY. Add it to `.env` and restart the backend.")
    top_k = st.slider("Snippets retrieved (top-k)", 1, 10, 5)
    m = info["models"]
    st.caption(
        f"Vision: `{m['vision']}`  \nAnswers: `{m['answer']}`  \n"
        f"Rewrite: `{m['rewrite']}`  \nEmbeddings: `{info['embed_model']}`"
    )

    st.subheader("Knowledge base")
    try:
        sources = api.list_sources()
    except api.APIError as e:
        sources = {}
        st.caption(f"Could not load sources: {e}")
    if not sources:
        st.caption("Empty — attach files in the chat box.")
    for name, n in sources.items():
        c1, c2 = st.columns([5, 1])
        c1.caption(f"{name} · {n} chunks")
        if c2.button("✕", key=f"del_{name}", help=f"Remove {name}"):
            api.delete_source(name)
            st.rerun()
    if sources and st.button("Clear knowledge base", use_container_width=True):
        api.clear_sources()
        st.rerun()
    if st.button("New chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

#-- time based greetings
from datetime import datetime
import streamlit.components.v1 as components
from zoneinfo import ZoneInfo


current_time = datetime.now(ZoneInfo("Asia/Kolkata"))
current_hour = current_time.hour


if 5 <= current_hour < 12:
    greeting = "Good Morning! Have a productive day!"
elif 12 <= current_hour < 17:
    greeting = "Good Afternoon! What can I help you with?"
elif 17 <= current_hour < 21:
    greeting = "Good Evening! Ready to explore somthing new?"
elif 21 <= current_hour < 24:
    greeting = "Good Night! Let's solve something."
else:
    greeting = "Hello, Night Owl"

# --- chat history

components.html(
    """
    <div class="title-container">
        <h1 class="animated-title">💬 QueryMind-AI</h1>
    </div>

    <style>
        .title-container {
            display: flex;
            justify-content: center;
            align-items: center;
            width: 100%;
            padding: 25px 0 10px 0;
        }

        .animated-title {
            color: #000000;
            font-size: 48px;
            font-weight: 750;
            text-align: center;
            font-family: Arial, sans-serif;
            margin: 0;
            animation: titleToggle 2s ease-in-out infinite;
        }

        @keyframes titleToggle {
            0%, 100% {
                opacity: 1;
                transform: scale(1);
            }

            50% {
                opacity: 0.65;
                transform: scale(1.04);
            }
        }
    </style>
    """,
    height=115,
)

# Your existing time-based greeting
st.markdown(
    f"""
    <div style="
        text-align: center;
        color: #000000;
        font-size: 17px;
        padding: 5px;
    ">
        {greeting}
    </div>
    """,
    unsafe_allow_html=True,
)


def show_sources(hits: list[dict]) -> None:
    with st.expander(f"Sources ({len(hits)})"):
        for i, h in enumerate(hits, start=1):
            meta = h["meta"]
            page = f" · p.{meta['page']}" if meta["page"] else ""
            st.markdown(f"**[{i}] {meta['source']}{page} · {meta['kind']}**")
            st.text(h["text"][:600] + ("…" if len(h["text"]) > 600 else ""))


def render_message(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        for _, img in msg.get("images", []):
            st.image(img, width=240)
        if msg.get("files"):
            st.markdown("".join(f"<span class='chip'>📄 {f}</span>" for f in msg["files"]), unsafe_allow_html=True)
        st.markdown(msg["content"])
        if msg.get("sources"):
            show_sources(msg["sources"])


for msg in st.session_state.messages:
    render_message(msg)

# chat input
prompt = st.chat_input(
    "Ask anything, or attach images, documents, tables…",
    accept_file="multiple",
    file_type=info["allowed_types"],
)

if prompt:
    question = (prompt.text or "").strip()
    uploads = list(prompt.files or [])
    if not question and uploads:
        question = "Summarize the attached file(s)."

    images, doc_files = [], []
    for f in uploads:
        ext = f.name.rsplit(".", 1)[-1].lower()
        (images if ext in IMAGE_TYPES else doc_files).append((f.name, f.getvalue()))

    user_msg = {"role": "user", "content": question, "images": images, "files": [n for n, _ in doc_files]}
    history = [{"role": x["role"], "content": x["content"]} for x in st.session_state.messages]
    st.session_state.messages.append(user_msg)
    render_message(user_msg)

    with st.chat_message("assistant"):
        try:
            # 1) send attached files to POST /ingest
            if uploads:
                with st.status("Reading your files…", expanded=True) as status:
                    results = api.ingest(images + doc_files)
                    failed = False
                    for r in results:
                        if r["error"] or r["chunks"] == 0:
                            failed = True
                            st.write(f"❌ {r['name']}: {r['error'] or 'no text could be extracted'}")
                        else:
                            st.write(f"✅ {r['name']} → {r['chunks']} chunks")
                    if failed:
                        status.update(label="Some files failed", state="error", expanded=True)
                    else:
                        status.update(label="Files processed", state="complete", expanded=False)
                if all(r["error"] or r["chunks"] == 0 for r in results):
                    st.stop()

            # 2) stream the answer from POST /chat
            stream = api.chat_stream(question, history, images, top_k)
            answer = st.write_stream(stream.tokens())
            if stream.sources:
                show_sources(stream.sources)
            st.session_state.messages.append({"role": "assistant", "content": answer, "sources": stream.sources})
        except api.APIError as e:
            st.error(f"Backend error — {e}")
            st.session_state.messages.append({"role": "assistant", "content": f" {e}"})
