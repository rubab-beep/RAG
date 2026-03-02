"""
config.py — Central configuration for Enterprise RAG
All tuneable parameters live here. Change nothing else.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM ──────────────────────────────────────────────────────────────────────
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
LLM_MODEL         = "gpt-4o-mini"          # Cost-effective, fast, accurate
LLM_TEMPERATURE   = 0.0                    # Deterministic — no creativity, no drift
LLM_MAX_TOKENS    = 1024

# ── Embeddings ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL   = "text-embedding-3-small"   # 1536-dim, cheap, strong
# FREE ALTERNATIVE (no API key needed):
# EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
USE_LOCAL_EMBEDDINGS = True               # Set True to use HuggingFace offline

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE        = 500      # ~120 words. Balances context richness vs precision
CHUNK_OVERLAP     = 75       # 15% overlap prevents cutting sentences mid-thought
CHUNK_SEPARATORS  = ["\n\n", "\n", ". ", " "]  # Respect paragraph > sentence > word

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K             = 4        # Retrieve 4 chunks. Sweet spot: enough context, low noise
SIMILARITY_METRIC = "cosine" # Best for semantic similarity with normalized embeddings
MIN_RELEVANCE_SCORE = 0.30   # Below this → treat as "not found". Range: 0.0–1.0

# Confidence thresholds (based on top chunk similarity score)
CONFIDENCE_HIGH   = 0.75     # Answer directly stated in documents
CONFIDENCE_MEDIUM = 0.50     # Answer implied / partially covered
# Below CONFIDENCE_MEDIUM → LOW confidence

# ── Vector Store ──────────────────────────────────────────────────────────────
VECTORSTORE_DIR   = "./vectorstore"
COLLECTION_NAME   = "company_knowledge"

# ── Paths ─────────────────────────────────────────────────────────────────────
DOCS_DIR          = "./data/documents"     # Drop PDFs here before running ingest.py

# ── UI ────────────────────────────────────────────────────────────────────────
APP_TITLE         = "Internal Knowledge Assistant"
APP_SUBTITLE      = "Answers strictly from company documents. No hallucinations."
