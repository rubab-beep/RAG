"""
query.py — Command-line interface for the RAG system.
Use this for testing, scripting, or when you don't want the Streamlit UI.

Usage:
    python query.py                       ← interactive loop
    python query.py --q "Your question"   ← single question
"""

import argparse
import sys

CONFIDENCE_ICONS = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}


def print_result(result: dict):
    conf = result["confidence"]
    icon = CONFIDENCE_ICONS.get(conf, "⚪")

    print("\n" + "─" * 65)
    print(result["answer"])
    print("─" * 65)
    print(f"\n{icon} Confidence: {conf}  |  Top similarity: {result['top_score']}")

    if result["sources"]:
        print("\n📄 Retrieved from:")
        for s in result["sources"]:
            print(f"   • {s['file']}  —  Page {s['page']}  (score: {s['relevance']})")
    print()


def interactive_loop(engine):
    print("\nType your question and press Enter. Type 'exit' to quit.\n")

    while True:
        try:
            question = input("❓ Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if question.lower() in ("exit", "quit", "q"):
            break

        if not question:
            continue

        result = engine.ask(question)
        print_result(result)


def main():
    parser = argparse.ArgumentParser(description="Enterprise RAG — CLI")
    parser.add_argument("--q", type=str, help="Single question (non-interactive mode)")
    args = parser.parse_args()

    try:
        from query_engine import RAGEngine
        engine = RAGEngine()
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Engine init failed: {e}")
        sys.exit(1)

    if args.q:
        result = engine.ask(args.q)
        print_result(result)
    else:
        interactive_loop(engine)


if __name__ == "__main__":
    main()
