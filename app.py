"""
app.py — Streamlit web interface for the Enterprise RAG system.
Run: streamlit run app.py

Two modes selectable via sidebar toggle:
  Answer Mode     — normal user Q&A (original behaviour, unchanged)
  Evaluation Mode — runs evaluation suite and shows metrics dashboard
"""

import os
import json
import shutil
import tempfile
import streamlit as st
from pathlib import Path

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Internal Knowledge Assistant",
    page_icon="🏢",
    layout="wide",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .confidence-high   { background:#d4edda; color:#155724; padding:4px 14px; border-radius:12px; font-weight:600; font-size:0.85rem; display:inline-block; }
    .confidence-medium { background:#fff3cd; color:#856404; padding:4px 14px; border-radius:12px; font-weight:600; font-size:0.85rem; display:inline-block; }
    .confidence-low    { background:#f8d7da; color:#721c24; padding:4px 14px; border-radius:12px; font-weight:600; font-size:0.85rem; display:inline-block; }
    .answer-box        { background:#fff; border:1px solid #e0e0e0; padding:18px 22px; border-radius:8px; line-height:1.8; white-space:pre-wrap; }
    .source-card       { background:#f8f9fa; border-left:3px solid #0d6efd; padding:8px 14px; margin:4px 0; border-radius:4px; font-size:0.85rem; }
    .doc-pill          { background:#e8f4fd; color:#0d6efd; padding:3px 10px; border-radius:10px; font-size:0.8rem; margin:2px; display:inline-block; }
    .metric-card       { background:#f8f9fa; border:1px solid #dee2e6; border-radius:8px; padding:16px; text-align:center; }
    .metric-value      { font-size:2rem; font-weight:700; color:#0d6efd; }
    .metric-label      { font-size:0.8rem; color:#666; margin-top:4px; }
    .pass-badge        { background:#d4edda; color:#155724; padding:2px 8px; border-radius:8px; font-size:0.75rem; }
    .fail-badge        { background:#f8d7da; color:#721c24; padding:2px 8px; border-radius:8px; font-size:0.75rem; }
</style>
""", unsafe_allow_html=True)


# ── Session State Defaults ────────────────────────────────────────────────────
for key, val in {
    "engine":        None,
    "history":       [],
    "indexed_files": [],
    "upload_dir":    None,
    "eval_report":   None,
    "app_mode":      "Answer Mode",
}.items():
    if key not in st.session_state:
        st.session_state[key] = val


# ── Helpers ───────────────────────────────────────────────────────────────────
def save_uploaded_files(uploaded_files) -> str:
    if st.session_state.upload_dir and os.path.exists(st.session_state.upload_dir):
        shutil.rmtree(st.session_state.upload_dir)
    upload_dir = tempfile.mkdtemp(prefix="rag_uploads_")
    st.session_state.upload_dir = upload_dir
    for f in uploaded_files:
        with open(os.path.join(upload_dir, f.name), "wb") as out:
            out.write(f.getbuffer())
    return upload_dir


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
    return RAGEngine(), len(pages), len(chunks)


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


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏢 Knowledge Assistant")
    st.divider()

    # Mode toggle — the key addition for evaluation
    st.markdown("### Mode")
    mode = st.radio(
        label      = "Select mode:",
        options    = ["Answer Mode", "Evaluation Mode"],
        index      = 0 if st.session_state.app_mode == "Answer Mode" else 1,
        label_visibility = "collapsed",
    )
    st.session_state.app_mode = mode

    st.divider()

    if st.session_state.indexed_files:
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
|---|---|
| LLM | `llama-3.1-8b` |
| Top-K | `{TOP_K}` |
| Chunk size | `{CHUNK_SIZE}` |
| Temp | `0.0` |
    """)

    if st.button("🗑️ Clear chat history"):
        st.session_state.history = []
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# UPLOAD SCREEN — shown when no engine loaded
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.engine is None:

    st.title("🏢 Internal Knowledge Assistant")
    st.caption("Upload documents · Ask questions · Evaluate performance")
    st.divider()

    st.markdown("""
    <div style="text-align:center;padding:20px 0 10px 0;">
        <h3>📂 Upload Your Documents to Get Started</h3>
        <p style="color:#666;">PDF · DOCX · TXT &nbsp;|&nbsp; Multiple files allowed</p>
    </div>""", unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        label="Drop files here or click to browse",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        cols = st.columns(3)
        for i, f in enumerate(uploaded_files):
            cols[i % 3].markdown(
                f'<span class="doc-pill">📄 {f.name} ({round(f.size/1024,1)} KB)</span>',
                unsafe_allow_html=True,
            )
        st.markdown("")

        if st.button(
            f"⚡ Process {len(uploaded_files)} document{'s' if len(uploaded_files)>1 else ''} and start",
            type="primary", use_container_width=True,
        ):
            progress = st.progress(0, text="Saving files...")
            try:
                st.session_state.engine = None
                st.cache_resource.clear()
                upload_dir = save_uploaded_files(uploaded_files)
                progress.progress(40, text="Embedding documents...")
                engine, n_pages, n_chunks = run_ingestion_on_uploads(upload_dir)
                progress.progress(95, text="Loading engine...")
                st.session_state.engine        = engine
                st.session_state.history       = []
                st.session_state.indexed_files = [f.name for f in uploaded_files]
                st.session_state.eval_report   = None
                progress.progress(100, text="Done!")
                st.success(f"✅ Ready! {len(uploaded_files)} file(s) · {n_pages} pages · {n_chunks} chunks")
                st.rerun()
            except Exception as e:
                progress.empty()
                st.error(f"❌ {e}")
    else:
        st.info("👆 Upload documents above to get started.")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# ANSWER MODE — original Q&A interface, completely unchanged
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.app_mode == "Answer Mode":

    st.title("🏢 Internal Knowledge Assistant")
    st.caption("Answers strictly from your documents · Zero hallucinations · Source cited on every response")

    # Active documents banner
    file_pills = " &nbsp;".join(
        f'<span class="doc-pill">📄 {f}</span>' for f in st.session_state.indexed_files
    )
    st.markdown(
        f'<div style="background:#f0f7ff;padding:10px 16px;border-radius:8px;margin-bottom:12px;">'
        f'<b>Active knowledge base:</b> &nbsp; {file_pills}</div>',
        unsafe_allow_html=True,
    )

    with st.form("question_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])
        with col1:
            question = st.text_input("Ask:", placeholder="What are the coffee processing methods?", label_visibility="collapsed")
        with col2:
            submitted = st.form_submit_button("Ask →", use_container_width=True, type="primary")

    if submitted and question.strip():
        with st.spinner("Searching documents..."):
            result = st.session_state.engine.ask(question)
        st.session_state.history.append({"question": question, "result": result})

    if st.session_state.history:
        render_result(st.session_state.history[-1]["question"], st.session_state.history[-1]["result"])
        if len(st.session_state.history) > 1:
            with st.expander(f"🕓 Previous questions ({len(st.session_state.history)-1})"):
                for item in reversed(st.session_state.history[:-1]):
                    render_result(item["question"], item["result"])
                    st.divider()
    else:
        st.markdown('<div style="text-align:center;color:#999;padding:40px 0;">💬 Ask your first question above</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION MODE — metrics dashboard
# ══════════════════════════════════════════════════════════════════════════════
else:

    st.title("📊 RAG Evaluation Dashboard")
    st.caption("Measures retrieval accuracy · Answer faithfulness · Refusal correctness")
    st.divider()

    col_run, col_opts = st.columns([3, 1])

    with col_opts:
        use_llm_faith = st.checkbox(
            "LLM faithfulness judge",
            value=False,
            help="Uses an extra LLM call per question for more accurate faithfulness scoring. Slower."
        )

    with col_run:
        run_eval = st.button(
            "▶️ Run Evaluation Suite",
            type="primary",
            use_container_width=True,
            help="Runs all questions from eval/ground_truth.py through the pipeline"
        )

    # ── Run evaluation ────────────────────────────────────────────────────────
    if run_eval:
        from eval.evaluator  import RAGEvaluator
        from eval.ground_truth import GROUND_TRUTH

        with st.spinner(f"Running {len(GROUND_TRUTH)} test questions through pipeline..."):
            evaluator  = RAGEvaluator(
                st.session_state.engine,
                use_llm_faithfulness=use_llm_faith
            )
            report     = evaluator.run(verbose=False)
            st.session_state.eval_report = report

        st.success(f"✅ Evaluation complete — {len(GROUND_TRUTH)} questions tested in {report['elapsed_seconds']}s")

    # ── Display report ────────────────────────────────────────────────────────
    report = st.session_state.eval_report

    if report is None:
        st.info("👆 Click **Run Evaluation Suite** to measure system performance.")

        st.markdown("### What gets measured?")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
**Retrieval Metrics**
- Recall@K
- Precision@K
- Mean Reciprocal Rank
            """)
        with col2:
            st.markdown("""
**Answer Quality**
- Keyword coverage
- Faithfulness score
- Hallucination rate
            """)
        with col3:
            st.markdown("""
**Safety Metrics**
- Refusal accuracy
- False positive rate
- Overall accuracy
            """)

        st.markdown("### Ground Truth Dataset")
        from eval.ground_truth import GROUND_TRUTH
        import pandas as pd
        df = pd.DataFrame([{
            "ID":         q["id"],
            "Question":   q["question"][:60] + "..." if len(q["question"]) > 60 else q["question"],
            "Category":   q.get("category", ""),
            "Difficulty": q.get("difficulty", ""),
            "Should Refuse": "✓" if q["should_refuse"] else "✗",
        } for q in GROUND_TRUTH])
        st.dataframe(df, use_container_width=True)

    else:
        # ── Top-level metric cards ────────────────────────────────
        st.markdown("### 📈 Aggregate Metrics")
        m1, m2, m3, m4, m5, m6 = st.columns(6)

        def metric_card(col, value, label, good_threshold=0.7, is_rate=False):
            display = f"{value*100:.1f}%" if is_rate else f"{value:.3f}"
            color   = "#28a745" if (value <= good_threshold if is_rate else value >= good_threshold) else "#dc3545"
            col.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value" style="color:{color}">{display}</div>'
                f'<div class="metric-label">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        metric_card(m1, report["recall_at_k"],          "Recall@K")
        metric_card(m2, report["precision_at_k"],        "Precision@K")
        metric_card(m3, report["mrr"],                   "MRR")
        metric_card(m4, report["avg_faithfulness"],      "Faithfulness")
        metric_card(m5, report["hallucination_rate"],    "Hallucination %", good_threshold=0.15, is_rate=True)
        metric_card(m6, report["refusal_overall_accuracy"], "Refusal Acc.")

        st.markdown("")

        # ── Grade ─────────────────────────────────────────────────
        recall = report["recall_at_k"]
        if   recall >= 0.85: grade, color = "✅ PRODUCTION READY",         "#28a745"
        elif recall >= 0.70: grade, color = "🟡 ACCEPTABLE — tune settings", "#ffc107"
        else:                grade, color = "🔴 NEEDS WORK",                 "#dc3545"

        st.markdown(
            f'<div style="background:{color}20;border:1px solid {color};'
            f'border-radius:8px;padding:12px 20px;margin:8px 0;">'
            f'<strong>System Grade: {grade}</strong></div>',
            unsafe_allow_html=True,
        )

        st.divider()

        # ── Charts ────────────────────────────────────────────────
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### Retrieval Metrics Breakdown")
            import pandas as pd

            retrieval_df = pd.DataFrame({
                "Metric": ["Recall@K", "Precision@K", "MRR", "Keyword Coverage", "Faithfulness"],
                "Score":  [
                    report["recall_at_k"],
                    report["precision_at_k"],
                    report["mrr"],
                    report["avg_keyword_coverage"],
                    report["avg_faithfulness"],
                ],
            })
            st.bar_chart(retrieval_df.set_index("Metric"))

        with col_right:
            st.markdown("#### Refusal Accuracy")
            refusal_df = pd.DataFrame({
                "Metric": ["Correct Refusals", "Wrong Refusals", "Overall Accuracy"],
                "Score":  [
                    report["refusal_true_pos_rate"],
                    report["refusal_false_pos_rate"],
                    report["refusal_overall_accuracy"],
                ],
            })
            st.bar_chart(refusal_df.set_index("Metric"))

        st.divider()

        # ── Per-question results table ────────────────────────────
        st.markdown("#### Per-Question Results")

        rows = []
        for pq in report.get("per_question", []):
            rows.append({
                "ID":           pq["id"],
                "Question":     pq["question"][:55] + "..." if len(pq["question"]) > 55 else pq["question"],
                "Category":     pq.get("category", ""),
                "Recall":       round(pq["recall"],           3),
                "Precision":    round(pq["precision"],         3),
                "MRR":          round(pq["rr"],                3),
                "Keyword Cov.": round(pq["keyword_coverage"],  3),
                "Faithfulness": round(pq["faithfulness"],      3),
                "Confidence":   pq["confidence"],
                "Pass":         "✅" if pq["passed"] else "❌",
            })

        if rows:
            results_df = pd.DataFrame(rows)
            st.dataframe(
                results_df,
                use_container_width=True,
                column_config={
                    "Pass":         st.column_config.TextColumn(width="small"),
                    "Recall":       st.column_config.ProgressColumn(min_value=0, max_value=1),
                    "Faithfulness": st.column_config.ProgressColumn(min_value=0, max_value=1),
                },
            )

        st.divider()

        # ── Saved reports ─────────────────────────────────────────
        st.markdown("#### 📁 Saved Evaluation Reports")
        results_dir = Path("./eval/results")
        if results_dir.exists():
            saved = sorted(results_dir.glob("eval_*.json"), reverse=True)
            if saved:
                for f in saved[:5]:
                    with open(f) as fp:
                        data = json.load(fp)
                    st.markdown(
                        f"- `{f.name}` — "
                        f"Recall: {data.get('recall_at_k', 0):.3f} | "
                        f"Faith: {data.get('avg_faithfulness', 0):.3f} | "
                        f"{data.get('timestamp', '')}"
                    )
            else:
                st.caption("No saved reports yet. Run evaluation to generate one.")

        # ── Recommendations ───────────────────────────────────────
        st.divider()
        st.markdown("#### 💡 Recommendations")

        recs = []
        if report["recall_at_k"] < 0.70:
            recs.append("📦 **Low Recall** — Try increasing `CHUNK_SIZE` to 700 or `TOP_K` to 6 in `config.py`")
        if report["precision_at_k"] < 0.50:
            recs.append("🎯 **Low Precision** — Too many irrelevant chunks. Lower `TOP_K` or raise `MIN_RELEVANCE_SCORE`")
        if report["avg_faithfulness"] < 0.70:
            recs.append("🔍 **Low Faithfulness** — LLM may be hallucinating. Review `prompts.py` grounding rules")
        if report["hallucination_rate"] > 0.20:
            recs.append("⚠️ **High Hallucination Rate** — Ensure `temperature=0.0` and context-only prompt rules are enforced")
        if report["refusal_false_pos_rate"] > 0.20:
            recs.append("🚫 **Too Many Wrong Refusals** — Lower `MIN_RELEVANCE_SCORE` in `config.py`")

        if recs:
            for rec in recs:
                st.markdown(rec)
        else:
            st.success("✅ All metrics within acceptable ranges. System is performing well.")
