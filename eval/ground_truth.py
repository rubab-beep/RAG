"""
eval/ground_truth.py — Static fallback ground truth dataset.

This file is used when no auto-generated ground truth exists yet.
Once you upload documents and the system auto-generates questions,
eval/generated_ground_truth.json takes over automatically.

You only need to edit this file if you want permanent hand-crafted tests.
"""

from typing import List, Dict, Any


# ── Static fallback dataset ───────────────────────────────────────────────────
# Replace these with questions from your own documents
# OR just leave them — auto-generation will override this anyway

GROUND_TRUTH: List[Dict[str, Any]] = [

    # Answerable questions — change these to match your documents
    {
        "id":               "Q01",
        "question":         "What is the main topic of the uploaded document?",
        "relevant_source":  "",        # fill in partial filename
        "relevant_page":    1,
        "answer_keywords":  [],        # fill in expected keywords
        "should_refuse":    False,
        "category":         "general",
        "difficulty":       "easy",
    },

    # Out-of-scope questions — these never change
    {
        "id":               "R01",
        "question":         "What is the current stock price of Apple?",
        "relevant_source":  None,
        "relevant_page":    None,
        "answer_keywords":  [],
        "should_refuse":    True,
        "category":         "out_of_scope",
        "difficulty":       "easy",
    },
    {
        "id":               "R02",
        "question":         "Who won the last FIFA World Cup?",
        "relevant_source":  None,
        "relevant_page":    None,
        "answer_keywords":  [],
        "should_refuse":    True,
        "category":         "out_of_scope",
        "difficulty":       "easy",
    },
    {
        "id":               "R03",
        "question":         "What is the population of Pakistan?",
        "relevant_source":  None,
        "relevant_page":    None,
        "answer_keywords":  [],
        "should_refuse":    True,
        "category":         "out_of_scope",
        "difficulty":       "easy",
    },
]


def get_answerable() -> List[Dict]:
    return [q for q in GROUND_TRUTH if not q["should_refuse"]]


def get_refusable() -> List[Dict]:
    return [q for q in GROUND_TRUTH if q["should_refuse"]]


def get_by_category(category: str) -> List[Dict]:
    return [q for q in GROUND_TRUTH if q.get("category") == category]


def get_by_difficulty(difficulty: str) -> List[Dict]:
    return [q for q in GROUND_TRUTH if q.get("difficulty") == difficulty]