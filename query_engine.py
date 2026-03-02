"""
query_engine.py — Core RAG query logic.
Used by both query.py (CLI) and app.py (Streamlit UI).

CHANGES FROM ORIGINAL:
  - ask()             : UNCHANGED — zero modifications to return contract
  - ask_with_trace()  : NEW — identical pipeline + returns raw chunks for eval
  - _run_pipeline()   : NEW — shared internal method called by both
"""

import os
from typing import Dict, Any

from langchain_groq  import ChatGroq
from langchain.schema import SystemMessage, HumanMessage

from prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, NO_CONTEXT_RESPONSE
from utils.retriever import (
    load_vectorstore,
    retrieve_chunks,
    score_to_confidence,
    format_context,
    format_sources,
)
from utils.embedder import get_embedding_model


class RAGEngine:
    """
    Single entry point for all RAG queries.
    Initialise once, call .ask() as many times as needed.

    Two public methods:
      ask()             -> Answer Mode   (normal user Q&A)
      ask_with_trace()  -> Evaluation Mode (same pipeline + raw internals)

    Both call _run_pipeline() internally — zero code duplication.
    """

    def __init__(self):
        print("Initialising RAG engine...")
        self.embedding_model = get_embedding_model()
        self.vectorstore     = load_vectorstore(self.embedding_model)
        self.llm             = ChatGroq(
            model       = "llama-3.1-8b-instant",
            temperature = 0.0,
            max_tokens  = 1024,
            api_key     = os.getenv("GROQ_API_KEY"),
        )
        print("✓ RAG engine ready\n")

    # ── PUBLIC: Answer Mode ───────────────────────────────────────
    def ask(self, question: str) -> Dict[str, Any]:
        """
        Full RAG pipeline for a single question.
        UNCHANGED from original implementation.

        Returns:
            answer, sources, confidence, top_score, found
        """
        result = self._run_pipeline(question)
        return {
            "answer":     result["answer"],
            "sources":    result["sources"],
            "confidence": result["confidence"],
            "top_score":  result["top_score"],
            "found":      result["found"],
        }

    # ── PUBLIC: Evaluation Mode ───────────────────────────────────
    def ask_with_trace(self, question: str) -> Dict[str, Any]:
        """
        Identical pipeline to ask() but also returns raw retrieval data.
        Used exclusively by the evaluation framework.

        Returns everything ask() returns PLUS:
            raw_chunks : list of chunk text strings
            raw_scores : list of similarity score floats
        """
        return self._run_pipeline(question)

    # ── PRIVATE: Shared Pipeline ──────────────────────────────────
    def _run_pipeline(self, question: str) -> Dict[str, Any]:
        """
        The actual RAG pipeline.
        Single source of truth — called by both ask() and ask_with_trace().
        """
        question = question.strip()
        if not question:
            return self._empty_result("Question cannot be empty.")

        # Step 1 — Retrieve
        chunks, scores, top_score = retrieve_chunks(question, self.vectorstore)
        confidence                = score_to_confidence(top_score, len(chunks))

        # Step 2 — Safety gate
        if not chunks:
            return {
                "answer":     NO_CONTEXT_RESPONSE,
                "sources":    [],
                "confidence": "LOW",
                "top_score":  0.0,
                "found":      False,
                "raw_chunks": [],
                "raw_scores": [],
            }

        # Step 3 — Build grounded prompt
        context_block = format_context(chunks, scores)
        system_msg    = SystemMessage(content=SYSTEM_PROMPT.format(context=context_block))
        user_msg      = HumanMessage(content=USER_PROMPT_TEMPLATE.format(question=question))

        # Step 4 — Call LLM
        response = self.llm.invoke([system_msg, user_msg])
        answer   = response.content.strip()

        # Step 5 — Format sources
        sources = format_sources(chunks, scores)

        return {
            "answer":     answer,
            "sources":    sources,
            "confidence": confidence,
            "top_score":  round(top_score, 3),
            "found":      True,
            "raw_chunks": [c.page_content for c in chunks],
            "raw_scores": [round(s, 4)    for s in scores],
        }

    @staticmethod
    def _empty_result(msg: str) -> Dict[str, Any]:
        return {
            "answer":     msg,
            "sources":    [],
            "confidence": "LOW",
            "top_score":  0.0,
            "found":      False,
            "raw_chunks": [],
            "raw_scores": [],
        }
