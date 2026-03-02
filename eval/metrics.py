"""
eval/metrics.py — Pure metric calculation functions.

Design principle:
  These are stateless functions only.
  They receive raw data and return numbers.
  Zero dependency on the RAG pipeline — fully testable in isolation.

Metrics implemented:
  - Recall@K          : Was the right chunk retrieved?
  - Precision@K       : What fraction of retrieved chunks were relevant?
  - Mean Reciprocal Rank (MRR) : How high was the correct chunk ranked?
  - Faithfulness      : Is the answer grounded in the retrieved context?
  - Hallucination Rate: How often does the LLM add unsupported facts?
  - Refusal Accuracy  : Does the system correctly refuse out-of-scope questions?
"""

import re
import os
from typing import List, Dict, Any, Optional


# ── Metric 1: Recall@K ────────────────────────────────────────────────────────

def recall_at_k(
    retrieved_sources: List[Dict],
    relevant_source:   str,
    relevant_page:     Optional[int] = None,
) -> float:
    """
    Recall@K: Was the relevant source retrieved in the top-K results?

    Returns 1.0 if found, 0.0 if not.
    Binary metric — either the right chunk was retrieved or it wasn't.

    Args:
        retrieved_sources: list of {"file": str, "page": int} dicts
        relevant_source:   partial filename to match against
        relevant_page:     expected page number (None = any page matches)
    """
    if not relevant_source:
        return 0.0

    for source in retrieved_sources:
        filename_match = relevant_source.lower() in source.get("file", "").lower()

        if relevant_page is not None:
            page_match = source.get("page") == relevant_page
            if filename_match and page_match:
                return 1.0
        else:
            if filename_match:
                return 1.0

    return 0.0


# ── Metric 2: Precision@K ─────────────────────────────────────────────────────

def precision_at_k(
    retrieved_sources: List[Dict],
    relevant_source:   str,
) -> float:
    """
    Precision@K: What fraction of retrieved chunks came from the relevant source?

    Example:
      Retrieved 4 chunks: 3 from relevant doc, 1 from irrelevant doc
      Precision@4 = 3/4 = 0.75

    Args:
        retrieved_sources: list of {"file": str, "page": int} dicts
        relevant_source:   partial filename to match against
    """
    if not retrieved_sources or not relevant_source:
        return 0.0

    relevant_count = sum(
        1 for s in retrieved_sources
        if relevant_source.lower() in s.get("file", "").lower()
    )

    return relevant_count / len(retrieved_sources)


# ── Metric 3: Mean Reciprocal Rank (MRR) ─────────────────────────────────────

def reciprocal_rank(
    retrieved_sources: List[Dict],
    relevant_source:   str,
    relevant_page:     Optional[int] = None,
) -> float:
    """
    Reciprocal Rank: 1/rank of the first relevant result.

    If relevant chunk is at rank 1: RR = 1.0  (best)
    If relevant chunk is at rank 2: RR = 0.5
    If relevant chunk is at rank 4: RR = 0.25
    If not found:                   RR = 0.0  (worst)

    MRR is the average of RR across all test cases.

    Args:
        retrieved_sources: list of {"file": str, "page": int} dicts
                           ordered from most to least similar
        relevant_source:   partial filename to match
        relevant_page:     expected page number (None = any page)
    """
    if not relevant_source:
        return 0.0

    for rank, source in enumerate(retrieved_sources, start=1):
        filename_match = relevant_source.lower() in source.get("file", "").lower()

        if relevant_page is not None:
            page_match = source.get("page") == relevant_page
            if filename_match and page_match:
                return 1.0 / rank
        else:
            if filename_match:
                return 1.0 / rank

    return 0.0


def mean_reciprocal_rank(rr_scores: List[float]) -> float:
    """Computes MRR from a list of individual reciprocal rank scores."""
    if not rr_scores:
        return 0.0
    return sum(rr_scores) / len(rr_scores)


# ── Metric 4: Answer Keyword Coverage ────────────────────────────────────────

def keyword_coverage(answer: str, expected_keywords: List[str]) -> float:
    """
    Checks what fraction of expected keywords appear in the answer.

    This is a lightweight proxy for answer relevance.
    Does not require a second LLM call.

    Returns:
        Float from 0.0 (no keywords found) to 1.0 (all keywords found)
    """
    if not expected_keywords:
        return 1.0   # no keywords to check = trivially satisfied

    answer_lower = answer.lower()
    found = sum(
        1 for kw in expected_keywords
        if kw.lower() in answer_lower
    )

    return found / len(expected_keywords)


# ── Metric 5: Faithfulness (Context Grounding) ───────────────────────────────

def faithfulness_score(
    answer:         str,
    context_chunks: List[str],
    llm_client=None,
) -> Dict[str, Any]:
    """
    Measures whether the answer is grounded in the retrieved context.

    Two modes:
      1. LLM-based (llm_client provided): asks LLM to audit the answer
         More accurate, costs one extra LLM call per question
      2. Heuristic (llm_client=None): checks word overlap
         Free, instant, less accurate

    Returns:
        {
          "score":    float (0.0 to 1.0),
          "method":   "llm" or "heuristic",
          "detail":   str explanation
        }
    """
    if llm_client is not None:
        return _faithfulness_llm(answer, context_chunks, llm_client)
    else:
        return _faithfulness_heuristic(answer, context_chunks)


def _faithfulness_llm(answer: str, context_chunks: List[str], llm_client) -> Dict:
    """LLM-based faithfulness — asks a judge LLM to evaluate grounding."""
    context_text = "\n\n".join(context_chunks[:4])   # limit context size

    prompt = f"""You are a faithfulness auditor for an AI system.

RETRIEVED CONTEXT:
{context_text}

ANSWER TO EVALUATE:
{answer}

Task: Identify every factual claim in the ANSWER.
For each claim, determine:
  SUPPORTED   = directly stated or clearly implied by CONTEXT
  UNSUPPORTED = not found in CONTEXT (hallucination risk)

Count the claims and return ONLY this JSON (no other text):
{{
  "total_claims": <int>,
  "supported_claims": <int>,
  "unsupported_claims": <int>,
  "faithfulness_score": <float 0.0-1.0>,
  "unsupported_examples": [<string>, ...]
}}"""

    try:
        from langchain.schema import HumanMessage
        response = llm_client.invoke([HumanMessage(content=prompt)])
        raw      = response.content.strip()

        # Strip markdown code fences if present
        raw = re.sub(r"```json|```", "", raw).strip()

        import json
        parsed = json.loads(raw)

        return {
            "score":  float(parsed.get("faithfulness_score", 0.5)),
            "method": "llm",
            "detail": f"Supported: {parsed.get('supported_claims')}/{parsed.get('total_claims')} claims",
            "unsupported": parsed.get("unsupported_examples", []),
        }

    except Exception as e:
        # Fallback to heuristic if LLM judge fails
        result         = _faithfulness_heuristic(answer, context_chunks)
        result["detail"] += f" (LLM judge failed: {e})"
        return result


def _faithfulness_heuristic(answer: str, context_chunks: List[str]) -> Dict:
    """
    Heuristic faithfulness using word overlap.

    Logic: if most content words in the answer also appear in the
    context, the answer is likely grounded.
    Crude but fast and free.
    """
    # Combine all context text
    context_text  = " ".join(context_chunks).lower()

    # Tokenize answer — remove stopwords, keep content words
    stopwords     = {
        "the","a","an","is","are","was","were","be","been","being",
        "have","has","had","do","does","did","will","would","could",
        "should","may","might","shall","can","it","its","this","that",
        "these","those","and","or","but","if","in","on","at","to","for",
        "of","with","by","from","as","into","through","during","before",
        "after","above","below","between","out","off","over","under","i",
        "you","he","she","we","they","them","their","what","which","who",
    }

    answer_words  = [
        w for w in re.findall(r'\b[a-z]+\b', answer.lower())
        if w not in stopwords and len(w) > 3
    ]

    if not answer_words:
        return {"score": 1.0, "method": "heuristic", "detail": "No content words to check"}

    grounded      = sum(1 for w in answer_words if w in context_text)
    score         = grounded / len(answer_words)

    return {
        "score":  round(score, 3),
        "method": "heuristic",
        "detail": f"{grounded}/{len(answer_words)} content words found in context",
    }


# ── Metric 6: Refusal Accuracy ────────────────────────────────────────────────

def refusal_accuracy(results: List[Dict]) -> Dict[str, float]:
    """
    Measures how accurately the system refuses out-of-scope questions.

    Computes:
      True Positive  Rate: correctly refused out-of-scope questions
      False Positive Rate: wrongly refused in-scope questions

    Args:
        results: list of eval result dicts with keys:
                 "should_refuse" (bool) and "found" (bool)
    """
    tp = fp = tn = fn = 0

    for r in results:
        should_refuse = r.get("should_refuse", False)
        was_refused   = not r.get("found", True)

        if should_refuse and was_refused:       tp += 1   # correct refusal
        elif not should_refuse and was_refused: fp += 1   # wrong refusal
        elif should_refuse and not was_refused: fn += 1   # missed refusal
        else:                                   tn += 1   # correct answer

    total_should_refuse = tp + fn
    total_should_answer = tn + fp

    return {
        "true_positive_rate":  tp / max(total_should_refuse, 1),
        "false_positive_rate": fp / max(total_should_answer, 1),
        "refusal_precision":   tp / max(tp + fp, 1),
        "overall_accuracy":    (tp + tn) / max(len(results), 1),
    }


# ── Metric 7: Hallucination Rate ─────────────────────────────────────────────

def hallucination_rate(faithfulness_scores: List[float], threshold: float = 0.70) -> float:
    """
    Estimates hallucination rate from faithfulness scores.

    A response is considered hallucinated if its faithfulness score
    drops below the threshold — meaning the answer contains claims
    not supported by the retrieved context.

    Args:
        faithfulness_scores: list of per-question faithfulness scores
        threshold:           below this = considered hallucinated (default 0.70)

    Returns:
        Float: fraction of responses considered hallucinated (0.0 = none)
    """
    if not faithfulness_scores:
        return 0.0

    hallucinated = sum(1 for s in faithfulness_scores if s < threshold)
    return hallucinated / len(faithfulness_scores)


# ── Aggregate Report ──────────────────────────────────────────────────────────

def build_report(eval_results: List[Dict]) -> Dict[str, Any]:
    """
    Computes all aggregate metrics from a list of per-question eval results.

    Each eval_result dict is produced by evaluator.py and contains:
      question, found, should_refuse, recall, precision, rr,
      keyword_coverage, faithfulness_score, sources, answer

    Returns a flat report dict suitable for JSON storage and UI display.
    """
    answerable = [r for r in eval_results if not r["should_refuse"]]
    refusable  = [r for r in eval_results if r["should_refuse"]]

    recall_scores      = [r["recall"]           for r in answerable if r.get("recall")           is not None]
    precision_scores   = [r["precision"]         for r in answerable if r.get("precision")         is not None]
    rr_scores          = [r["rr"]                for r in answerable if r.get("rr")                is not None]
    kw_scores          = [r["keyword_coverage"]  for r in answerable if r.get("keyword_coverage")  is not None]
    faith_scores       = [r["faithfulness"]      for r in eval_results if r.get("faithfulness")    is not None]
    refusal_stats      = refusal_accuracy(eval_results)

    def avg(lst): return round(sum(lst) / len(lst), 3) if lst else 0.0

    return {
        "total_questions":         len(eval_results),
        "answerable_questions":    len(answerable),
        "refusable_questions":     len(refusable),

        # Retrieval metrics (on answerable questions only)
        "recall_at_k":             avg(recall_scores),
        "precision_at_k":          avg(precision_scores),
        "mrr":                     avg(rr_scores),

        # Answer quality metrics
        "avg_keyword_coverage":    avg(kw_scores),
        "avg_faithfulness":        avg(faith_scores),
        "hallucination_rate":      round(hallucination_rate(faith_scores), 3),

        # Refusal metrics
        "refusal_true_pos_rate":   round(refusal_stats["true_positive_rate"],  3),
        "refusal_false_pos_rate":  round(refusal_stats["false_positive_rate"], 3),
        "refusal_overall_accuracy":round(refusal_stats["overall_accuracy"],    3),

        # Score breakdown
        "recall_scores":           recall_scores,
        "precision_scores":        precision_scores,
        "rr_scores":               rr_scores,
        "faithfulness_scores":     faith_scores,
    }
