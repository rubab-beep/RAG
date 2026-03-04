"""
query_engine.py — Core RAG query logic.
"""

import os
from typing import Dict, Any, List

from langchain_groq   import ChatGroq
from langchain.schema import SystemMessage, HumanMessage, Document

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

    def __init__(self):
        print("Initialising RAG engine...")
        self.embedding_model = get_embedding_model()
        self.vectorstore     = load_vectorstore(self.embedding_model)
        self.all_chunks      = self._load_all_chunks()
        self.llm             = ChatGroq(
            model       = "llama-3.1-8b-instant",
            temperature = 0.0,
            max_tokens  = 2048,
            api_key     = os.getenv("GROQ_API_KEY"),
        )
        print("✓ RAG engine ready\n")


    def _load_all_chunks(self) -> List:
        """Loads all chunks from vectorstore for neighbour retrieval."""
        try:
            collection = self.vectorstore._collection
            data       = collection.get(include=["documents", "metadatas"])
            chunks     = []

            for doc_text, meta in zip(data["documents"], data["metadatas"]):
                chunks.append(Document(
                    page_content = doc_text,
                    metadata     = meta or {}
                ))

            chunks.sort(key=lambda c: (
                c.metadata.get("source", ""),
                c.metadata.get("page", 0),
                c.metadata.get("chunk_index", 0)
            ))

            print(f"  Loaded {len(chunks)} chunks for retrieval")
            return chunks

        except Exception as e:
            print(f"  ⚠️  Could not load all chunks: {e}")
            return []


    def ask(self, question: str) -> Dict[str, Any]:
        """Answer Mode — normal user Q&A."""
        result = self._run_pipeline(question)
        return {
            "answer":     result["answer"],
            "sources":    result["sources"],
            "confidence": result["confidence"],
            "top_score":  result["top_score"],
            "found":      result["found"],
        }


    def ask_with_trace(self, question: str) -> Dict[str, Any]:
        """Evaluation Mode — same pipeline + raw internals for metrics."""
        return self._run_pipeline(question)


    def ask_streaming(self, question: str):
        """Streams LLM response token by token."""
        question = question.strip()
        if not question:
            yield "Question cannot be empty."
            return

        chunks, scores, top_score = retrieve_chunks(question, self.vectorstore)

        if not chunks:
            yield NO_CONTEXT_RESPONSE
            return

        context_block = format_context(chunks, scores)
        system_msg    = SystemMessage(content=SYSTEM_PROMPT.format(context=context_block))
        user_msg      = HumanMessage(content=USER_PROMPT_TEMPLATE.format(question=question))

        for chunk in self.llm.stream([system_msg, user_msg]):
            yield chunk.content


    def _run_pipeline(self, question: str) -> Dict[str, Any]:
        """
        Single shared pipeline called by ask() and ask_with_trace().
        """
        question = question.strip()
        if not question:
            return self._empty_result("Question cannot be empty.")

        # ── Step 1: Retrieve ──────────────────────────────────────
        try:
            from utils.retriever import retrieve_with_neighbours
            if self.all_chunks:
                chunks, scores, top_score = retrieve_with_neighbours(
                    question,
                    self.vectorstore,
                    self.all_chunks,
                )
            else:
                chunks, scores, top_score = retrieve_chunks(
                    question, self.vectorstore
                )
        except (ImportError, Exception):
            chunks, scores, top_score = retrieve_chunks(
                question, self.vectorstore
            )

        confidence = score_to_confidence(top_score, len(chunks))

        # ── Step 2: Safety gate ───────────────────────────────────
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

        # ── Step 3: Build prompt ──────────────────────────────────
        context_block = format_context(chunks, scores)
        system_msg    = SystemMessage(
            content=SYSTEM_PROMPT.format(context=context_block)
        )
        user_msg = HumanMessage(
            content=USER_PROMPT_TEMPLATE.format(question=question)
        )

        # ── Step 4: Call LLM ──────────────────────────────────────
        response = self.llm.invoke([system_msg, user_msg])
        answer   = response.content.strip()

        # ── Step 5: Format sources ────────────────────────────────
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