"""
utils/embedder.py — Embedding model factory.
Supports OpenAI (default) or local HuggingFace (free, offline).
Switch via config.py: USE_LOCAL_EMBEDDINGS = True
"""

from config import EMBEDDING_MODEL, USE_LOCAL_EMBEDDINGS, OPENAI_API_KEY


def get_embedding_model():
    """
    Returns the appropriate embedding model based on config.
    OpenAI: Requires OPENAI_API_KEY. Fast, high quality.
    Local:  No API key. Runs on CPU. Slightly lower quality but free.
    """
    if USE_LOCAL_EMBEDDINGS:
        return _get_local_embeddings()
    else:
        return _get_openai_embeddings()


def _get_openai_embeddings():
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError:
        raise ImportError("Run: pip install langchain-openai")

    if not OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY not set. Either:\n"
            "  1. Add it to your .env file\n"
            "  2. Set USE_LOCAL_EMBEDDINGS=True in config.py for free local embeddings"
        )

    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_key=OPENAI_API_KEY,
    )


def _get_local_embeddings():
    """
    Uses sentence-transformers — runs fully offline, no API key needed.
    Model: all-MiniLM-L6-v2 (22M params, 384-dim, fast on CPU)
    """
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        raise ImportError("Run: pip install langchain-community sentence-transformers")

    print("  Using local HuggingFace embeddings (no API key required)")
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},  # Required for cosine similarity
    )
