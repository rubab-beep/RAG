"""
utils/retriever.py — Vector store operations: build, load, search.
Uses ChromaDB for persistence. Cosine similarity. Returns chunks + scores.
"""

import os
from typing import List, Tuple, Optional
from langchain.schema import Document
from langchain_community.vectorstores import Chroma

from config import (
    VECTORSTORE_DIR,
    COLLECTION_NAME,
    TOP_K,
    MIN_RELEVANCE_SCORE,
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
)

def build_vectorstore(chunks: List[Document], embedding_model) -> Chroma:
    import time
    import shutil

    if os.path.exists(VECTORSTORE_DIR):
        for attempt in range(6):
            try:
                shutil.rmtree(VECTORSTORE_DIR)
                break
            except PermissionError:
                if attempt == 5:
                    print(f"  ⚠️  Could not delete old store, overwriting in place...")
                    break
                print(f"  Waiting for file lock... ({attempt + 1}/5)")
                time.sleep(1.5)

    print(f"  Embedding {len(chunks)} chunks — this may take a minute...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=COLLECTION_NAME,
        persist_directory=VECTORSTORE_DIR,
        collection_metadata={"hnsw:space": "cosine"},
    )

    print(f"  ✓ Vector store saved to {VECTORSTORE_DIR}")
    return vectorstore


def load_vectorstore(embedding_model) -> Chroma:
    """
    Loads a previously built vector store from disk.
    Raises a clear error if ingest.py hasn't been run yet.
    """
    if not os.path.exists(VECTORSTORE_DIR):
        raise FileNotFoundError(
            f"Vector store not found at '{VECTORSTORE_DIR}'.\n"
            f"Run: python ingest.py\n"
            f"Make sure your documents are in: ./data/documents/"
        )

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=VECTORSTORE_DIR,
        collection_metadata={"hnsw:space": "cosine"},
    )


def retrieve_chunks(
    question: str,
    vectorstore: Chroma,
    k: int = TOP_K,
) -> Tuple[List[Document], List[float], float]:
    """
    Searches the vector store for the most relevant chunks.

    Returns:
        chunks         — List of matching Document objects
        scores         — Parallel list of similarity scores (0.0–1.0)
        top_score      — Highest similarity score (used for confidence level)
    """
    # similarity_search_with_relevance_scores returns (doc, score) tuples
    # Score is cosine similarity: 1.0 = identical, 0.0 = unrelated
    results = vectorstore.similarity_search_with_relevance_scores(question, k=k)

    # Filter out results below minimum relevance threshold
    filtered = [(doc, score) for doc, score in results if score >= MIN_RELEVANCE_SCORE]

    if not filtered:
        return [], [], 0.0

    chunks = [doc for doc, _ in filtered]
    scores = [score for _, score in filtered]
    top_score = max(scores)

    return chunks, scores, top_score


def score_to_confidence(top_score: float, chunks_found: int) -> str:
    """
    Converts top similarity score into a human-readable confidence label.
    """
    if chunks_found == 0:
        return "LOW"
    if top_score >= CONFIDENCE_HIGH:
        return "HIGH"
    elif top_score >= CONFIDENCE_MEDIUM:
        return "MEDIUM"
    else:
        return "LOW"


def format_context(chunks: List[Document], scores: List[float]) -> str:
    """
    Assembles retrieved chunks into a formatted context block for the prompt.
    Includes source metadata inline so the LLM can cite accurately.
    """
    parts = []
    for i, (chunk, score) in enumerate(zip(chunks, scores), start=1):
        meta = chunk.metadata
        header = (
            f"[EXCERPT {i}] "
            f"Source: {meta.get('source', 'Unknown')} | "
            f"Page: {meta.get('page', '?')} | "
            f"Relevance: {score:.2f}"
        )
        parts.append(f"{header}\n{chunk.page_content}")

    return "\n\n" + "─" * 60 + "\n\n".join(parts) + "\n\n" + "─" * 60


def format_sources(chunks: List[Document], scores: List[float]) -> List[dict]:
    """
    Returns a clean list of source metadata dicts for display in the UI.
    """
    seen = set()
    sources = []

    for chunk, score in zip(chunks, scores):
        meta = chunk.metadata
        key = (meta.get("source"), meta.get("page"))
        if key not in seen:
            seen.add(key)
            sources.append({
                "file": meta.get("source", "Unknown"),
                "page": meta.get("page", "?"),
                "relevance": round(score, 3),
            })

    return sources
