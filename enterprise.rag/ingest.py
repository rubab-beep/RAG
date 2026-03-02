"""
ingest.py — Offline document ingestion pipeline.
Run this ONCE (and re-run when documents change).
It will NOT run automatically when the app starts.

Usage:
    python ingest.py
    python ingest.py --docs ./path/to/my/documents

What it does:
    1. Loads all PDFs/DOCX/TXT from ./data/documents/
    2. Extracts text with page-level metadata
    3. Splits into overlapping chunks
    4. Embeds every chunk using the configured model
    5. Persists to ChromaDB vector store on disk
"""

import argparse
import sys
import time

from config import DOCS_DIR
from utils.loader import load_documents
from utils.chunker import chunk_pages
from utils.embedder import get_embedding_model
from utils.retriever import build_vectorstore


def run_ingestion(docs_dir: str = DOCS_DIR):
    print("=" * 60)
    print("  Enterprise RAG — Document Ingestion Pipeline")
    print("=" * 60)

    start = time.time()

    # ── Step 1: Load documents ────────────────────────────────────
    print("\n[1/4] Loading documents...")
    try:
        pages = load_documents(docs_dir)
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

    # ── Step 2: Chunk ─────────────────────────────────────────────
    print("\n[2/4] Chunking pages...")
    chunks = chunk_pages(pages)

    if not chunks:
        print("❌ No text content found in documents. Check your files.")
        sys.exit(1)

    # ── Step 3: Load embedding model ──────────────────────────────
    print("\n[3/4] Loading embedding model...")
    try:
        embedding_model = get_embedding_model()
        print("  ✓ Embedding model ready")
    except (ImportError, ValueError) as e:
        print(f"\n❌ Embedding model error: {e}")
        sys.exit(1)

    # ── Step 4: Build vector store ────────────────────────────────
    print("\n[4/4] Building vector store...")
    try:
        vectorstore = build_vectorstore(chunks, embedding_model)
    except Exception as e:
        print(f"\n❌ Vector store error: {e}")
        sys.exit(1)

    elapsed = time.time() - start

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ✅ Ingestion complete")
    print(f"  Documents processed : {len(set(p['source'] for p in pages))}")
    print(f"  Pages extracted     : {len(pages)}")
    print(f"  Chunks created      : {len(chunks)}")
    print(f"  Time elapsed        : {elapsed:.1f}s")
    print("=" * 60)
    print("\nYou can now run:")
    print("  python query.py                 ← CLI interface")
    print("  streamlit run app.py            ← Web interface")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG document ingestion")
    parser.add_argument(
        "--docs",
        default=DOCS_DIR,
        help=f"Path to documents directory (default: {DOCS_DIR})",
    )
    args = parser.parse_args()
    run_ingestion(args.docs)
