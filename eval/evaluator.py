"""
eval/evaluator.py — Evaluation orchestrator.

Design:
  Uses the EXISTING RAGEngine.ask_with_trace() method (added below).
  Does NOT touch retrieval, chunking, or prompts.
  Inserts evaluation hooks AFTER the pipeline runs, not inside it.

Two execution modes share identical pipeline code:
  Answer Mode:     engine.ask(question)          → returns answer dict
  Evaluation Mode: evaluator.run(ground_truth)   → returns metrics report
"""

import os
import json
import time
from typing  import List, Dict, Any, Optional
from pathlib import Path

from eval.ground_truth import GROUND_TRUTH, get_answerable, get_refusable
from eval.metrics      import (
    recall_at_k,
    precision_at_k,
    reciprocal_rank,
    keyword_coverage,
    faithfulness_score,
    build_report,
)


class RAGEvaluator:
    """
    Wraps the existing RAGEngine and runs systematic evaluation.

    Usage:
        from query_engine  import RAGEngine
        from eval.evaluator import RAGEvaluator

        engine    = RAGEngine()                    # existing engine
        evaluator = RAGEvaluator(engine)           # wrap it
        report    = evaluator.run()                # run all tests
    """

    RESULTS_DIR = "./eval/results"

    def __init__(self, engine, use_llm_faithfulness: bool = False):
        """
        Args:
            engine:                 existing RAGEngine instance
            use_llm_faithfulness:   True = use LLM to judge faithfulness
                                    False = use fast heuristic (default)
                                    LLM mode costs one extra API call per question
        """
        self.engine                = engine
        self.use_llm_faithfulness  = use_llm_faithfulness
        os.makedirs(self.RESULTS_DIR, exist_ok=True)


    def run(
        self,
        dataset:      Optional[List[Dict]] = None,
        label:        str                  = "default",
        verbose:      bool                 = True,
    ) -> Dict[str, Any]:
        """
        Runs evaluation over the full ground truth dataset.

        Args:
            dataset:  list of ground truth dicts (defaults to GROUND_TRUTH)
            label:    name for this run — used in saved JSON filename
            verbose:  print progress to terminal

        Returns:
            Full evaluation report dict including per-question results
            and aggregate metrics.
        """
        dataset       = dataset or GROUND_TRUTH
        per_question  = []
        start_time    = time.time()

        if verbose:
            print(f"\n{'='*60}")
            print(f"  RAG EVALUATION RUN — {label.upper()}")
            print(f"  {len(dataset)} questions | "
                  f"{len(get_answerable())} answerable | "
                  f"{len(get_refusable())} refusable")
            print(f"{'='*60}\n")

        for i, case in enumerate(dataset, start=1):
            q_result = self._evaluate_one(case, verbose=verbose)
            per_question.append(q_result)

            if verbose:
                status = "✅" if q_result["passed"] else "❌"
                print(
                    f"  {status} [{case['id']}] "
                    f"Recall:{q_result['recall']:.2f} "
                    f"Faith:{q_result['faithfulness']:.2f} "
                    f"KW:{q_result['keyword_coverage']:.2f}"
                )

        # ── Aggregate all metrics ─────────────────────────────────
        report = build_report(per_question)
        report["label"]              = label
        report["elapsed_seconds"]    = round(time.time() - start_time, 1)
        report["per_question"]       = per_question
        report["timestamp"]          = time.strftime("%Y-%m-%d %H:%M:%S")

        # ── Save to JSON ──────────────────────────────────────────
        self._save_report(report, label)

        if verbose:
            self._print_summary(report)

        return report


    def _evaluate_one(self, case: Dict, verbose: bool = False) -> Dict[str, Any]:
        """
        Runs one ground truth question through the pipeline and
        computes all metrics for it.

        This is where evaluation hooks are inserted into the existing flow.
        The engine.ask_with_trace() call runs the IDENTICAL pipeline as
        engine.ask() — the only difference is it also returns raw chunks.
        """
        question       = case["question"]
        relevant_src   = case.get("relevant_source")
        relevant_page  = case.get("relevant_page")
        expected_kw    = case.get("answer_keywords", [])
        should_refuse  = case.get("should_refuse", False)

        # ── HOOK: call existing pipeline with trace enabled ───────
        trace = self.engine.ask_with_trace(question)

        answer         = trace["answer"]
        found          = trace["found"]
        sources        = trace["sources"]
        top_score      = trace["top_score"]
        raw_chunks     = trace["raw_chunks"]       # list of chunk texts
        raw_scores     = trace["raw_scores"]       # parallel similarity scores

        # ── Compute metrics ───────────────────────────────────────
        rec  = recall_at_k(sources, relevant_src, relevant_page) \
               if not should_refuse else None

        prec = precision_at_k(sources, relevant_src) \
               if not should_refuse else None

        rr   = reciprocal_rank(sources, relevant_src, relevant_page) \
               if not should_refuse else None

        kw   = keyword_coverage(answer, expected_kw) \
               if not should_refuse else None

        faith_result = faithfulness_score(
            answer         = answer,
            context_chunks = raw_chunks,
            llm_client     = self.engine.llm if self.use_llm_faithfulness else None,
        )

        # ── Determine pass/fail ───────────────────────────────────
        if should_refuse:
            passed = not found            # correct if it refused
        else:
            passed = (
                (rec or 0)  >= 0.5 and
                (kw  or 0)  >= 0.5 and
                faith_result["score"] >= 0.5
            )

        return {
            "id":               case["id"],
            "question":         question,
            "category":         case.get("category", ""),
            "difficulty":       case.get("difficulty", ""),
            "should_refuse":    should_refuse,
            "found":            found,
            "answer":           answer[:300],    # truncate for storage
            "top_score":        top_score,
            "confidence":       trace["confidence"],
            "recall":           round(rec,  3) if rec  is not None else 0.0,
            "precision":        round(prec, 3) if prec is not None else 0.0,
            "rr":               round(rr,   3) if rr   is not None else 0.0,
            "keyword_coverage": round(kw,   3) if kw   is not None else 1.0,
            "faithfulness":     round(faith_result["score"], 3),
            "faith_method":     faith_result["method"],
            "faith_detail":     faith_result["detail"],
            "sources":          sources,
            "passed":           passed,
        }


    def _save_report(self, report: Dict, label: str):
        """Saves the report to a timestamped JSON file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename  = f"eval_{label}_{timestamp}.json"
        path      = Path(self.RESULTS_DIR) / filename

        # Save without per_question raw chunks to keep file small
        save_data = {k: v for k, v in report.items() if k != "per_question"}
        save_data["per_question"] = [
            {k: v for k, v in pq.items() if k not in ("answer",)}
            for pq in report.get("per_question", [])
        ]

        with open(path, "w") as f:
            json.dump(save_data, f, indent=2)

        print(f"\n  📁 Report saved: {path}")


    def _print_summary(self, report: Dict):
        """Prints a formatted summary table to terminal."""
        print(f"\n{'='*60}")
        print(f"  EVALUATION SUMMARY — {report.get('label', '').upper()}")
        print(f"{'='*60}")
        print(f"  Questions tested  : {report['total_questions']}")
        print(f"  Time elapsed      : {report['elapsed_seconds']}s")
        print(f"{'─'*60}")
        print(f"  RETRIEVAL METRICS")
        print(f"    Recall@K        : {report['recall_at_k']:.3f}")
        print(f"    Precision@K     : {report['precision_at_k']:.3f}")
        print(f"    MRR             : {report['mrr']:.3f}")
        print(f"{'─'*60}")
        print(f"  ANSWER QUALITY")
        print(f"    Keyword Coverage: {report['avg_keyword_coverage']:.3f}")
        print(f"    Faithfulness    : {report['avg_faithfulness']:.3f}")
        print(f"    Hallucination % : {report['hallucination_rate']*100:.1f}%")
        print(f"{'─'*60}")
        print(f"  REFUSAL ACCURACY")
        print(f"    Correct refusals: {report['refusal_true_pos_rate']:.3f}")
        print(f"    Wrong refusals  : {report['refusal_false_pos_rate']:.3f}")
        print(f"    Overall accuracy: {report['refusal_overall_accuracy']:.3f}")
        print(f"{'='*60}")

        # Grade
        recall = report["recall_at_k"]
        if recall >= 0.85:
            grade = "✅ PRODUCTION READY"
        elif recall >= 0.70:
            grade = "🟡 ACCEPTABLE — tune settings"
        else:
            grade = "🔴 NEEDS WORK — check chunking and embeddings"

        print(f"\n  GRADE: {grade}")
        print(f"{'='*60}\n")


    def load_results(self) -> List[Dict]:
        """Loads all saved evaluation reports from disk."""
        results = []
        results_dir = Path(self.RESULTS_DIR)

        if not results_dir.exists():
            return []

        for f in sorted(results_dir.glob("eval_*.json"), reverse=True):
            try:
                with open(f) as fp:
                    results.append(json.load(fp))
            except Exception:
                pass

        return results
