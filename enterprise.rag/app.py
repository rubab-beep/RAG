"""
app.py — Streamlit web interface for the Enterprise RAG system.
Run: streamlit run app.py

Flow:
  1. User uploads PDFs/DOCX/TXT directly from the browser
  2. Files are saved to a temp folder and ingested automatically
  3. Q&A interface unlocks once documents are indexed
  4. User can upload new documents at any time to reset
"""

import os
import shutil
import tempfile
import streamlit as st

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Internal Knowledge Assistant",
    page_icon="🏢",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .confidence-high   { background:#d4edda; color:#155724; padding:4px 14px; border-radius:12px; font-weight:600; font-size:0.85rem; display:inline-block; }
    .confidence-medium { background:#fff3cd; color:#856404; padding:4px 14px; border-radius:12px; font-weight:600; font-size:0.85rem; display:inline-block; }
    .confidence-low    { background:#f8d7da; color:#721c24; padding:4px 14px; border-radius:12px; font-weight:600; font-size:0.85rem; display:inline-block; }
    .answer-box  { background:#ffffff; border:1px solid #e0e0e0; padding:18px 22px; border-radius:8px; line-height:1.8; white-space:pre-wrap; }
    .source-card { background:#f8f9fa; border-left:3px solid #0d6efd; padding:8px 14px; margin:4px 0; border-radius:4px; font-size:0.85rem; }
    .doc-pill    { background:#e8f4fd; color:#0d6efd; padding:3px 10px; border-radius:10px; font-size:0.8rem; margin:2px; display:inline-block; }
    .block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)


# ── Session State Defaults ────────────────────────────────────────────────────
if "engine"        not in st.session_state: st.session_state.engine        = None
if "history"       not in st.session_state: st.session_state.history       = []
if "indexed_files" not in st.session_state: st.session_state.indexed_files = []
if "upload_dir"    not in st.session_state: st.session_state.upload_dir    = None


# ── Helper: Save uploaded files to temp dir ───────────────────────────────────
def save_uploaded_files(uploaded_files) -> str:
    if st.session_state.upload_dir and os.path.exists(st.session_state.upload_dir):
        shutil.rmtree(st.session_state.upload_dir)
    upload_dir = tempfile.mkdtemp(prefix="rag_uploads_")
    st.session_state.upload_dir = upload_dir
    for f in uploaded_files:
        dest = os.path.join(upload_dir, f.name)
        with open(dest, "wb") as out:
            out.write(f.getbuffer())
    return upload_dir


# ── Helper: Full ingestion + engine init ─────────────────────────────────────
def run_ingestion_on_uploads(upload_dir: str):
    from utils.loader   import load_documents
    from utils.chunker  import chunk_pages
    from utils.embedder import get_embedding_model
    from utils.retriever import build_vectorstore
    from query_engine   import RAGEngine

    pages  = load_documents(upload_dir)
    chunks = chunk_pages(pages)
    if not chunks:
        raise ValueError("No text could be extracted from the uploaded files.")
    embedding_model = get_embedding_model()
    build_vectorstore(chunks, embedding_model)
    engine = RAGEngine()
    return engine, len(pages), len(chunks)


# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏢 Internal Knowledge Assistant")
st.caption("Upload your company documents · Ask questions · Get sourced, grounded answers")
st.divider()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 — UPLOAD SCREEN
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.engine is None:

    st.markdown("""
    <div style="text-align:center; padding:20px 0 10px 0;">
        <h3>📂 Upload Your Documents to Get Started</h3>
        <p style="color:#666;">Supported formats: PDF &nbsp;·&nbsp; DOCX &nbsp;·&nbsp; TXT &nbsp;|&nbsp; Multiple files allowed</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        label="Drop files here or click to browse",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        st.markdown("**Files selected:**")
        cols = st.columns(3)
        for i, f in enumerate(uploaded_files):
            size_kb = round(f.size / 1024, 1)
            cols[i % 3].markdown(
                f'<span class="doc-pill">📄 {f.name} ({size_kb} KB)</span>',
                unsafe_allow_html=True,
            )

        st.markdown("")
        if st.button(
            f"⚡ Process {len(uploaded_files)} document{'s' if len(uploaded_files) > 1 else ''} and start",
            type="primary",
            use_container_width=True,
        ):
            progress = st.progress(0, text="Saving uploaded files...")
            try:
                upload_dir = save_uploaded_files(uploaded_files)
                 # Release old ChromaDB connection before rebuilding
                st.session_state.engine = None
                st.cache_resource.clear()

                upload_dir = save_uploaded_files(uploaded_files)
                progress.progress(25, text="Extracting text from documents...")
                progress.progress(45, text="Chunking and embedding — this takes ~30 seconds...")
                engine, n_pages, n_chunks = run_ingestion_on_uploads(upload_dir)
                progress.progress(95, text="Loading knowledge base...")
                st.session_state.engine        = engine
                st.session_state.history       = []
                st.session_state.indexed_files = [f.name for f in uploaded_files]
                progress.progress(100, text="Done!")
                st.success(
                    f"✅ **Ready!** Indexed {len(uploaded_files)} file(s) · "
                    f"{n_pages} pages · {n_chunks} chunks"
                )
                st.rerun()
            except Exception as e:
                progress.empty()
                st.error(f"❌ Processing failed: {e}")
    else:
        st.info("👆 Upload one or more company documents above. Once processed, you can ask questions about their contents.")

    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — Q&A INTERFACE
# ══════════════════════════════════════════════════════════════════════════════
engine = st.session_state.engine

# Active documents banner
file_pills = " &nbsp;".join(
    f'<span class="doc-pill">📄 {f}</span>' for f in st.session_state.indexed_files
)
st.markdown(
    f'<div style="background:#f0f7ff;padding:10px 16px;border-radius:8px;margin-bottom:12px;">'
    f'<b>Active knowledge base:</b> &nbsp; {file_pills}</div>',
    unsafe_allow_html=True,
)

# Question input
with st.form("question_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    with col1:
        question = st.text_input(
            "Ask a question:",
            placeholder="e.g. What are the coffee processing methods?",
            label_visibility="collapsed",
        )
    with col2:
        submitted = st.form_submit_button("Ask →", use_container_width=True, type="primary")

# Process query
if submitted and question.strip():
    with st.spinner("Searching documents..."):
        result = engine.ask(question)
    st.session_state.history.append({"question": question, "result": result})


# Render a single result
def render_result(q: str, result: dict):
    conf       = result["confidence"]
    conf_class = f"confidence-{conf.lower()}"
    conf_label = {"HIGH": "🟢 HIGH", "MEDIUM": "🟡 MEDIUM", "LOW": "🔴 LOW"}.get(conf, conf)

    st.markdown(f"**Q: {q}**")
    st.markdown(
        f'<span class="{conf_class}">Confidence: {conf_label}</span> '
        f'<span style="color:#999;font-size:0.8rem;margin-left:8px;">score: {result["top_score"]}</span>',
        unsafe_allow_html=True,
    )
    st.markdown("")
    st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)

    if result["sources"]:
        st.markdown("**📄 Sources**")
        for s in result["sources"]:
            st.markdown(
                f'<div class="source-card">📋 <b>{s["file"]}</b>'
                f' &nbsp;·&nbsp; Page {s["page"]}'
                f' &nbsp;·&nbsp; Relevance: {s["relevance"]}</div>',
                unsafe_allow_html=True,
            )
    st.markdown("")


# Display results
if st.session_state.history:
    render_result(
        st.session_state.history[-1]["question"],
        st.session_state.history[-1]["result"],
    )
    if len(st.session_state.history) > 1:
        with st.expander(f"🕓 Previous questions ({len(st.session_state.history) - 1})"):
            for item in reversed(st.session_state.history[:-1]):
                render_result(item["question"], item["result"])
                st.divider()
else:
    st.markdown(
        '<div style="text-align:center;color:#999;padding:40px 0;">💬 Ask your first question above</div>',
        unsafe_allow_html=True,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📂 Loaded Documents")
    for f in st.session_state.indexed_files:
        st.markdown(f"- {f}")

    st.divider()

    if st.button("🔄 Upload new documents", use_container_width=True):
        st.session_state.engine        = None
        st.session_state.history       = []
        st.session_state.indexed_files = []
        if st.session_state.upload_dir and os.path.exists(st.session_state.upload_dir):
            shutil.rmtree(st.session_state.upload_dir)
        st.session_state.upload_dir = None
        st.rerun()

    st.divider()
    st.markdown("### ℹ️ System")
    from config import TOP_K, CHUNK_SIZE
    st.markdown(f"""
| Setting | Value |
|---------|-------|
| LLM | `llama-3.1-8b` |
| Chunks retrieved | `{TOP_K}` |
| Chunk size | `{CHUNK_SIZE} chars` |
| Temperature | `0.0` |
    """)

    st.divider()
    st.markdown("**How it works:**")
    st.markdown("""
1. Your files are chunked and embedded locally
2. Your question is matched against chunks
3. LLM answers **only** from matched chunks
4. No match found → honest refusal
    """)

    if st.button("🗑️ Clear chat history"):
        st.session_state.history = []
        st.rerun()