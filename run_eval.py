"""
run_eval.py — Standalone CLI evaluation runner.

Usage:
    python run_eval.py                        # run full eval suite
    python run_eval.py --mode experiments     # run chunk/embedding experiments
    python run_eval.py --llm-faith            # use LLM faithfulness judge
    python run_eval.py --verbose              # detailed per-question output

This script requires:
  - Documents already uploaded and vectorstore built (run the app first)
  - GROQ_API_KEY set in .env
"""

import argparse
import sys
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# FIXED — checks folder exists and has any files at all
vectorstore_path = Path("./vectorstore")
if not vectorstore_path.exists() or not any(vectorstore_path.rglob("*")):
    print("❌ No vectorstore found.")
    print("   Launch the app first: streamlit run app.py")
    print("   Upload documents, then come back and run this script.")
    sys.exit(1)


def run_evaluation(use_llm_faith: bool = False, verbose: bool = True):
    from query_engine    import RAGEngine
    from eval.evaluator  import RAGEvaluator
    from eval.ground_truth import GROUND_TRUTH

    print(f"Loading RAG engine...")
    engine    = RAGEngine()
    evaluator = RAGEvaluator(engine, use_llm_faithfulness=use_llm_faith)

    print(f"Running {len(GROUND_TRUTH)} test questions...")
    report = evaluator.run(verbose=verbose)
    return report


def run_experiments():
    from eval.experiments import run_chunk_experiments, save_comparison

    docs_dir = "./data/documents"
    if not Path(docs_dir).exists() or not any(Path(docs_dir).iterdir()):
        print("❌ No documents in ./data/documents/")
        print("   Copy your PDFs there or upload via the app first.")
        sys.exit(1)

    print("Running chunk size experiments...")
    results = run_chunk_experiments(docs_dir)
    save_comparison(results, label="chunk_experiments")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Evaluation CLI")
    parser.add_argument(
        "--mode",
        choices=["eval", "experiments"],
        default="eval",
        help="eval = run ground truth suite | experiments = test multiple configs"
    )
    parser.add_argument(
        "--llm-faith",
        action="store_true",
        help="Use LLM judge for faithfulness (slower, more accurate)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Print per-question results"
    )

    args = parser.parse_args()

    if args.mode == "eval":
        run_evaluation(use_llm_faith=args.llm_faith, verbose=args.verbose)
    elif args.mode == "experiments":
        run_experiments()
