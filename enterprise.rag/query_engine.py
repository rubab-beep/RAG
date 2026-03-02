"""
query_engine.py — Core RAG query logic.
"""

import os
from typing import Dict, Any
from langchain_groq import ChatGroq
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

    def __init__(self):
        print("Initialising RAG engine...")
        self.embedding_model = get_embedding_model()
        self.vectorstore = load_vectorstore(self.embedding_model)
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0.0,
            max_tokens=1024,
            api_key=os.getenv("GROQ_API_KEY"),
        )
        print("✓ RAG engine ready\n")

    def ask(self, question: str) -> Dict[str, Any]:
        question = question.strip()
        if not question:
            return self._empty_result("Question cannot be empty.")

        chunks, scores, top_score = retrieve_chunks(question, self.vectorstore)
        confidence = score_to_confidence(top_score, len(chunks))

        if not chunks:
            return {
                "answer": NO_CONTEXT_RESPONSE,
                "sources": [],
                "confidence": "LOW",
                "top_score": 0.0,
                "found": False,
            }

        context_block = format_context(chunks, scores)
        system_msg = SystemMessage(content=SYSTEM_PROMPT.format(context=context_block))
        user_msg = HumanMessage(
            content=USER_PROMPT_TEMPLATE.format(question=question)
        )

        response = self.llm.invoke([system_msg, user_msg])
        answer = response.content.strip()
        sources = format_sources(chunks, scores)

        return {
            "answer": answer,
            "sources": sources,
            "confidence": confidence,
            "top_score": round(top_score, 3),
            "found": True,
        }

    @staticmethod
    def _empty_result(msg: str) -> Dict[str, Any]:
        return {
            "answer": msg,
            "sources": [],
            "confidence": "LOW",
            "top_score": 0.0,
            "found": False,
        }