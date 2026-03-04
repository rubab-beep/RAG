"""
app.py — Streamlit web interface for the Enterprise RAG system.
Run: streamlit run app.py
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
    .metric-card { background:#ffffff; border:1px solid #e0e0e0; padding:16px; border-radius:8px; text-align:center; }
    .metric-value-good { font-size:2rem; font-weight:700; color:#28a745; }
    .metric-value-bad  { font-size:2rem; font-weight:700; color:#dc3545; }
    .metric-label { font-size:0.85rem; color:#666; margin-top:4px; }
</style>
""", unsafe_allow_html=True)


# ── Session State Defaults ────────────────────────────────────────────────────
if "engine"        not in st.session_state: st.session_state.engine        = None
if "history"       not in st.session_state: st.session_state.history       = []
if "indexed_files" not in st.session_state: st.session_state.indexed_files = []
if "upload_dir"    not in st.session_state: st.session_state.upload_dir    = None
if "app_mode"      not in st.session_state: st.session_state.app_mode      = "Answer Mode"
if "eval_report"   not in st.session_state: st.session_state.eval_report   = None


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
    from utils.loader    import load_documents
    from utils.chunker   import chunk_pages
    from utils.embedder  import get_embedding_model
    from utils.retriever import build_vectorstore
    from query_engine    import RAGEngine

    pages  = load_documents(upload_dir)
    chunks = chunk_pages(pages)
    if not chunks:
        raise ValueError("No text could be extracted from the uploaded files.")
    embedding_model = get_embedding_model()
    build_vectorstore(chunks, embedding_model)
    engine = RAGEngine()

    # Auto-generate ground truth after ingestion
    try:
        from eval.generate_ground_truth import generate_ground_truth
        llm = getattr(engine, 'llm', None)
        if llm:
            generate_ground_truth(
                chunks            = chunks,
                llm_client        = llm,
                questions_per_doc = 5,
                save_to_file      = True,
            )
    except Exception as e:
        print(f"  ⚠️  Ground truth generation failed: {e}")

    return engine, len(pages), len(chunks)


# ── Helper: Render single Q&A result ─────────────────────────────────────────
def render_result(q: str, result: dict):
    conf       = result["confidence"]
    conf_clean = conf.replace("🟢 ", "").replace("🟡 ", "").replace("🔴 ", "").upper()
    conf_class = {"HIGH": "confidence-high", "MEDIUM": "confidence-medium"}.get(conf_clean, "confidence-low")

    st.markdown(f"**Q: {q}**")
    st.markdown(
        f'<span class="{conf_class}">Confidence: {conf}</span> '
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


# ── Helper: Metric card ───────────────────────────────────────────────────────
def metric_card(col, value, label, good_threshold=0.70, is_rate=False):
    if is_rate:
        display  = f"{value*100:.1f}%"
        is_good  = value <= good_threshold
    else:
        display  = f"{value:.3f}"
        is_good  = value >= good_threshold

    css_class = "metric-value-good" if is_good else "metric-value-bad"
    col.markdown(
        f'<div class="metric-card">'
        f'<div class="{css_class}">{display}</div>'
        f'<div class="metric-label">{label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


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
                st.session_state.eval_report   = None
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
        st.info("👆 Upload one or more documents above. Once processed, you can ask questions about their contents.")

    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2 — MAIN INTERFACE (Answer Mode + Evaluation Mode)
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


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:

    # Mode toggle
    st.markdown("### 🔧 Mode")
    mode = st.radio(
        label   = "Select mode:",
        options = ["Answer Mode", "Evaluation Mode"],
        index   = 0,
    )
    st.session_state.app_mode = mode

    st.divider()
    st.markdown("### 📂 Loaded Documents")
    for f in st.session_state.indexed_files:
        st.markdown(f"- {f}")

    st.divider()

    if st.button("🔄 Upload new documents", use_container_width=True):
        st.session_state.engine        = None
        st.session_state.history       = []
        st.session_state.indexed_files = []
        st.session_state.eval_report   = None
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


# ══════════════════════════════════════════════════════════════════════════════
# ANSWER MODE
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.app_mode == "Answer Mode":

    with st.form("question_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            question = st.text_input(
                "Ask a question:",
                placeholder="e.g. What are the main findings of the paper?",
                label_visibility="collapsed",
            )
        with col2:
            submitted = st.form_submit_button("Ask →", use_container_width=True, type="primary")

    if submitted and question.strip():
        with st.spinner("Searching documents..."):
            result = engine.ask(question)
        st.session_state.history.append({"question": question, "result": result})

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


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION MODE
# ══════════════════════════════════════════════════════════════════════════════
else:

    st.markdown("## 📊 Evaluation Dashboard")
    st.caption("Measures retrieval accuracy, answer quality, and refusal behaviour.")

    # Run evaluation button
    if st.button("▶️ Run Evaluation Suite", type="primary", use_container_width=True):
        try:
            from eval.evaluator import RAGEvaluator

            with st.spinner("Running evaluation — this may take 1-2 minutes..."):
                evaluator = RAGEvaluator(engine)
                report    = evaluator.run(verbose=False)
                st.session_state.eval_report = report

            st.success(f"✅ Evaluation complete — {report['total_questions']} questions tested in {report['elapsed_seconds']}s")

        except Exception as e:
            st.error(f"❌ Evaluation failed: {e}")
            st.exception(e)

    report = st.session_state.eval_report

    # ── No report yet ─────────────────────────────────────────
    if report is None:
        st.info("👆 Click **Run Evaluation Suite** to test your RAG system automatically.")

        try:
            from eval.generate_ground_truth import load_generated_ground_truth
            gt = load_generated_ground_truth()
            if gt:
                st.markdown(f"**Auto-generated test dataset:** {len(gt)} questions ready")
                answerable = [q for q in gt if not q.get("should_refuse")]
                refusable  = [q for q in gt if q.get("should_refuse")]
                st.markdown(f"- {len(answerable)} answerable questions")
                st.markdown(f"- {len(refusable)} refusal questions")
                with st.expander("Preview test questions"):
                    for q in gt[:5]:
                        icon = "✅" if not q.get("should_refuse") else "🚫"
                        st.markdown(f"{icon} **[{q['id']}]** {q['question']}")
            else:
                st.warning("No ground truth found. Re-upload documents to auto-generate test questions.")
        except Exception:
            pass

    # ── Show report ───────────────────────────────────────────
    else:
        st.divider()
        st.markdown("### 📈 Aggregate Metrics")

        # Six metric cards
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        metric_card(c1, report["recall_at_k"],          "Recall@K",        good_threshold=0.70)
        metric_card(c2, report["precision_at_k"],       "Precision@K",     good_threshold=0.60)
        metric_card(c3, report["mrr"],                  "MRR",             good_threshold=0.70)
        metric_card(c4, report["avg_faithfulness"],     "Faithfulness",    good_threshold=0.50)
        metric_card(c5, report["hallucination_rate"],   "Hallucination %", good_threshold=0.25, is_rate=True)
        metric_card(c6, report.get("refusal_overall_accuracy", 0), "Refusal Acc.", good_threshold=0.70)

        st.markdown("")

        # System grade
        recall = report["recall_at_k"]
        if   recall >= 0.85: grade = "✅ PRODUCTION READY";  grade_color = "#d4edda"
        elif recall >= 0.70: grade = "🟡 ACCEPTABLE";        grade_color = "#fff3cd"
        else:                grade = "🔴 NEEDS WORK";        grade_color = "#f8d7da"

        st.markdown(
            f'<div style="background:{grade_color};padding:12px 20px;border-radius:8px;font-weight:600;">'
            f'System Grade: {grade}</div>',
            unsafe_allow_html=True,
        )

        st.divider()

        # Charts
        try:
            import plotly.graph_objects as go

            col_left, col_right = st.columns(2)

            with col_left:
                st.markdown("**Retrieval Metrics Breakdown**")
                fig1 = go.Figure(go.Bar(
                    x = ["Faithfulness", "Keyword Coverage", "MRR", "Precision@K", "Recall@K"],
                    y = [
                        report["avg_faithfulness"],
                        sum(pq.get("keyword_coverage", 0) for pq in report.get("per_question", [])) / max(len(report.get("per_question", [])), 1),
                        report["mrr"],
                        report["precision_at_k"],
                        report["recall_at_k"],
                    ],
                    marker_color = "#1f77b4",
                ))
                fig1.update_layout(yaxis_range=[0, 1], height=300, margin=dict(t=10, b=60))
                st.plotly_chart(fig1, use_container_width=True)

            with col_right:
                st.markdown("**Refusal Accuracy**")
                fig2 = go.Figure(go.Bar(
                    x = ["Correct Refusals", "Overall Accuracy", "Wrong Refusals"],
                    y = [
                        report.get("refusal_true_pos_rate", 0),
                        report.get("refusal_overall_accuracy", 0),
                        report.get("refusal_false_pos_rate", 0),
                    ],
                    marker_color = "#1f77b4",
                ))
                fig2.update_layout(yaxis_range=[0, 1], height=300, margin=dict(t=10, b=60))
                st.plotly_chart(fig2, use_container_width=True)

        except ImportError:
            st.info("Install plotly for charts: pip install plotly")

        st.divider()

        # Per-question results table
        st.markdown("### 🔍 Per-Question Results")

        per_q = report.get("per_question", [])
        if per_q:
            for pq in per_q:
                passed  = pq.get("passed", False)
                icon    = "✅" if passed else "❌"
                refuse  = pq.get("should_refuse", False)
                q_type  = "🚫 Refusal" if refuse else "💬 Answer"

                with st.expander(f"{icon} [{pq['id']}] {pq['question'][:80]}"):
                    col_a, col_b, col_c, col_d = st.columns(4)
                    col_a.metric("Recall",      f"{pq.get('recall', 0):.2f}")
                    col_b.metric("Faithfulness",f"{pq.get('faithfulness', 0):.2f}")
                    col_c.metric("Keyword Cov.",f"{pq.get('keyword_coverage', 0):.2f}")
                    col_d.metric("Type", q_type)

                    if pq.get("answer"):
                        st.markdown("**Answer preview:**")
                        st.markdown(f"> {pq['answer'][:300]}...")

        st.divider()

        # Recommendations
        st.markdown("### 💡 Recommendations")
        recs = []

        if report["recall_at_k"] < 0.70:
            recs.append("🔧 **Low Recall** — Increase `TOP_K` or `CHUNK_SIZE` in config.py")
        if report["precision_at_k"] < 0.50:
            recs.append("🔧 **Low Precision** — Raise `MIN_RELEVANCE_SCORE` in config.py")
        if report["avg_faithfulness"] < 0.50:
            recs.append("🔧 **Low Faithfulness** — Review prompts.py rules, confirm temperature=0.0")
        if report["hallucination_rate"] > 0.25:
            recs.append("🔧 **High Hallucination** — Add stricter grounding rules to prompts.py")
        if report.get("refusal_false_pos_rate", 0) > 0.20:
            recs.append("🔧 **Too many wrong refusals** — Lower `MIN_RELEVANCE_SCORE` in config.py")
        if report.get("refusal_true_pos_rate", 0) < 0.50:
            recs.append("🔧 **Missing refusals** — Raise `MIN_RELEVANCE_SCORE` in config.py")

        if recs:
            for r in recs:
                st.markdown(r)
        else:
            st.success("✅ All metrics look healthy — system is performing well!")

        # Saved reports
        st.divider()
        st.markdown("### 📁 Saved Reports")
        try:
            from eval.evaluator import RAGEvaluator
            evaluator   = RAGEvaluator(engine)
            all_reports = evaluator.load_results()
            if all_reports:
                for r in all_reports[:5]:
                    st.markdown(
                        f"- **{r.get('label','?')}** — "
                        f"{r.get('timestamp','?')} — "
                        f"Recall: {r.get('recall_at_k',0):.2f} — "
                        f"Faith: {r.get('avg_faithfulness',0):.2f}"
                    )
            else:
                st.caption("No saved reports yet.")
        except Exception:
            pass