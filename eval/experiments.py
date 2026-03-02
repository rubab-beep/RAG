"""
eval/experiments.py — Chunking and embedding experiment runner.

Runs the evaluation suite across multiple configurations and
produces a comparison JSON showing which settings perform best.

Uses the SAME ingestion pipeline — only config values change.
No new pipelines. No duplicate code.

Usage:
    python -m eval.experiments

Output:
    eval/results/experiment_comparison_<timestamp>.json
"""

import os
import sys
import json
import time
import shutil
from typing  import List, Dict, Any
from pathlib import Path


# Experiment configurations to test
# Add or remove entries to control what gets tested
CHUNK_EXPERIMENTS = [
    {"chunk_size": 300,  "chunk_overlap": 50,  "label": "small_chunks"},
    {"chunk_size": 500,  "chunk_overlap": 75,  "label": "default"},
    {"chunk_size": 700,  "chunk_overlap": 100, "label": "medium_chunks"},
    {"chunk_size": 1000, "chunk_overlap": 150, "label": "large_chunks"},
]

EMBEDDING_EXPERIMENTS = [
    {
        "use_local":    True,
        "model_name":   "sentence-transformers/all-MiniLM-L6-v2",
        "label":        "minilm_384",
    },
    {
        "use_local":    True,
        "model_name":   "sentence-transformers/all-mpnet-base-v2",
        "label":        "mpnet_768",
    },
    # Uncomment to include OpenAI embeddings (requires API key + credits)
    # {
    #     "use_local":  False,
    #     "model_name": "text-embedding-3-small",
    #     "label":      "openai_3small_1536",
    # },
]


def run_chunk_experiments(upload_dir: str) -> List[Dict]:
    """
    Re-ingests the same documents with different chunk sizes.
    Runs full evaluation after each ingestion.
    Returns list of reports for comparison.
    """
    from utils.loader    import load_documents
    from utils.embedder  import get_embedding_model
    from utils.retriever import build_vectorstore

    # Patch config at runtime — import config module directly
    import config

    pages           = load_documents(upload_dir)
    embedding_model = get_embedding_model()
    results         = []

    print("\n" + "="*60)
    print("  CHUNK SIZE EXPERIMENT")
    print("="*60)

    for exp in CHUNK_EXPERIMENTS:
        print(f"\n  Testing: {exp['label']} "
              f"(size={exp['chunk_size']}, overlap={exp['chunk_overlap']})")

        # Temporarily override config values
        original_size    = config.CHUNK_SIZE
        original_overlap = config.CHUNK_OVERLAP

        config.CHUNK_SIZE    = exp["chunk_size"]
        config.CHUNK_OVERLAP = exp["chunk_overlap"]

        try:
            # Re-chunk with new settings using the SAME pages
            from utils.chunker import chunk_pages
            import importlib
            import utils.chunker
            importlib.reload(utils.chunker)      # reload to pick up new config
            from utils.chunker import chunk_pages

            chunks = chunk_pages(pages)
            build_vectorstore(chunks, embedding_model)

            # Run evaluation
            from query_engine   import RAGEngine
            from eval.evaluator import RAGEvaluator

            engine    = RAGEngine()
            evaluator = RAGEvaluator(engine)
            report    = evaluator.run(label=exp["label"], verbose=False)
            report["experiment_type"]   = "chunk_size"
            report["chunk_size"]        = exp["chunk_size"]
            report["chunk_overlap"]     = exp["chunk_overlap"]
            report["total_chunks"]      = len(chunks)

            results.append(report)

            print(f"    Recall@K:     {report['recall_at_k']:.3f}")
            print(f"    MRR:          {report['mrr']:.3f}")
            print(f"    Faithfulness: {report['avg_faithfulness']:.3f}")
            print(f"    Chunks:       {len(chunks)}")

        except Exception as e:
            print(f"    ❌ Failed: {e}")

        finally:
            # Always restore original config
            config.CHUNK_SIZE    = original_size
            config.CHUNK_OVERLAP = original_overlap

    return results


def run_embedding_experiments(upload_dir: str) -> List[Dict]:
    """
    Re-ingests documents with different embedding models.
    Requires models to be installed.
    """
    import config
    from utils.loader    import load_documents
    from utils.chunker   import chunk_pages
    from utils.retriever import build_vectorstore

    pages   = load_documents(upload_dir)
    chunks  = chunk_pages(pages)
    results = []

    print("\n" + "="*60)
    print("  EMBEDDING MODEL EXPERIMENT")
    print("="*60)

    for exp in EMBEDDING_EXPERIMENTS:
        print(f"\n  Testing: {exp['label']} ({exp['model_name']})")

        original_model  = config.EMBEDDING_MODEL
        original_local  = config.USE_LOCAL_EMBEDDINGS

        config.EMBEDDING_MODEL    = exp["model_name"]
        config.USE_LOCAL_EMBEDDINGS = exp["use_local"]

        try:
            import utils.embedder
            import importlib
            importlib.reload(utils.embedder)
            from utils.embedder import get_embedding_model

            embedding_model = get_embedding_model()
            build_vectorstore(chunks, embedding_model)

            from query_engine   import RAGEngine
            from eval.evaluator import RAGEvaluator

            engine    = RAGEngine()
            evaluator = RAGEvaluator(engine)
            report    = evaluator.run(label=exp["label"], verbose=False)
            report["experiment_type"]  = "embedding_model"
            report["embedding_model"]  = exp["model_name"]
            report["embedding_dims"]   = 768 if "mpnet" in exp["model_name"] else 384

            results.append(report)

            print(f"    Recall@K:     {report['recall_at_k']:.3f}")
            print(f"    MRR:          {report['mrr']:.3f}")
            print(f"    Faithfulness: {report['avg_faithfulness']:.3f}")

        except Exception as e:
            print(f"    ❌ Failed: {e}")

        finally:
            config.EMBEDDING_MODEL      = original_model
            config.USE_LOCAL_EMBEDDINGS = original_local

    return results


def save_comparison(all_results: List[Dict], label: str = "comparison"):
    """Saves a side-by-side comparison of all experiment results."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path      = Path("./eval/results") / f"experiment_{label}_{timestamp}.json"

    os.makedirs("./eval/results", exist_ok=True)

    # Build comparison table — strip per_question detail for brevity
    comparison = []
    for r in all_results:
        comparison.append({
            "label":                r.get("label"),
            "experiment_type":      r.get("experiment_type"),
            "chunk_size":           r.get("chunk_size"),
            "chunk_overlap":        r.get("chunk_overlap"),
            "embedding_model":      r.get("embedding_model"),
            "recall_at_k":          r.get("recall_at_k"),
            "precision_at_k":       r.get("precision_at_k"),
            "mrr":                  r.get("mrr"),
            "avg_keyword_coverage": r.get("avg_keyword_coverage"),
            "avg_faithfulness":     r.get("avg_faithfulness"),
            "hallucination_rate":   r.get("hallucination_rate"),
            "refusal_overall_accuracy": r.get("refusal_overall_accuracy"),
        })

    with open(path, "w") as f:
        json.dump({"runs": comparison, "timestamp": timestamp}, f, indent=2)

    print(f"\n  📊 Comparison saved: {path}")
    _print_comparison_table(comparison)

    return path


def _print_comparison_table(comparison: List[Dict]):
    """Prints a formatted comparison table to terminal."""
    print(f"\n{'='*75}")
    print(f"  EXPERIMENT COMPARISON")
    print(f"{'─'*75}")
    print(f"  {'Label':<20} {'Recall':>7} {'Prec':>7} {'MRR':>7} {'Faith':>7} {'Halluc':>7}")
    print(f"{'─'*75}")

    for r in comparison:
        print(
            f"  {r['label']:<20} "
            f"{r['recall_at_k'] or 0:>7.3f} "
            f"{r['precision_at_k'] or 0:>7.3f} "
            f"{r['mrr'] or 0:>7.3f} "
            f"{r['avg_faithfulness'] or 0:>7.3f} "
            f"{(r['hallucination_rate'] or 0)*100:>6.1f}%"
        )

    print(f"{'='*75}")

    # Highlight best configuration
    best = max(comparison, key=lambda x: (x.get("recall_at_k") or 0))
    print(f"\n  🏆 Best config by Recall@K: {best['label']}")


if __name__ == "__main__":
    # Quick standalone run — uses existing vectorstore documents dir
    docs_dir = "./data/documents"

    if not Path(docs_dir).exists() or not list(Path(docs_dir).iterdir()):
        print("❌ No documents found in ./data/documents/")
        print(" Upload documents first via the web UI, then run experiments.")
        sys.exit(1)

    all_results = []
    all_results.extend(run_chunk_experiments(docs_dir))

    save_comparison(all_results, label="chunk_experiments")
