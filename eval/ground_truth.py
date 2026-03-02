"""
eval/ground_truth.py — Ground truth dataset for RAG evaluation.

HOW TO USE:
  Edit GROUND_TRUTH below to match YOUR uploaded documents.
  Each entry needs:
    - question       : the test question
    - relevant_source: filename (partial match is fine)
    - relevant_page  : page number where answer lives
    - answer_keywords: words that MUST appear in a correct answer
    - should_refuse  : True if this question has no answer in docs

IMPORTANT:
  This is the ONLY file you edit when adding new documents.
  All metrics are computed automatically from these entries.
"""

from typing import List, Dict, Any

GROUND_TRUTH = [

    # Questions FROM your actual documents
    {
        "id":               "Q01",
        "question":         "What is penetration testing?",
        "relevant_source":  "Penetration",   # partial filename is fine
        "relevant_page":    1,
        "answer_keywords":  ["security", "vulnerability", "testing", "system"],
        "should_refuse":    False,
        "category":         "security",
        "difficulty":       "easy",
    },
    {
        "id":               "Q02",
        "question":         "What types of AI are discussed?",
        "relevant_source":  "AI",
        "relevant_page":    1,
        "answer_keywords":  ["machine", "learning", "neural", "model"],
        "should_refuse":    False,
        "category":         "ai_concepts",
        "difficulty":       "easy",
    },
    {
        "id":               "Q03",
        "question":         "What cars are in the factsheet?",
        "relevant_source":  "factsheet_cars",
        "relevant_page":    1,
        "answer_keywords":  [],   # fill in after reading your document
        "should_refuse":    False,
        "category":         "vehicles",
        "difficulty":       "easy",
    },

    # Out-of-scope questions — should be refused
    {
        "id":               "R01",
        "question":         "What is the population of Pakistan?",
        "relevant_source":  None,
        "relevant_page":    None,
        "answer_keywords":  [],
        "should_refuse":    True,
        "category":         "out_of_scope",
        "difficulty":       "easy",
    },
    {
        "id":               "R02",
        "question":         "Who won the last cricket World Cup?",
        "relevant_source":  None,
        "relevant_page":    None,
        "answer_keywords":  [],
        "should_refuse":    True,
        "category":         "out_of_scope",
        "difficulty":       "easy",
    },
]

def get_answerable() -> List[Dict]:
    """Returns only questions that should be answered (not refused)."""
    return [q for q in GROUND_TRUTH if not q["should_refuse"]]


def get_refusable() -> List[Dict]:
    """Returns only questions that should trigger a refusal."""
    return [q for q in GROUND_TRUTH if q["should_refuse"]]


def get_by_category(category: str) -> List[Dict]:
    """Returns questions filtered by category."""
    return [q for q in GROUND_TRUTH if q.get("category") == category]


def get_by_difficulty(difficulty: str) -> List[Dict]:
    """Returns questions filtered by difficulty level."""
    return [q for q in GROUND_TRUTH if q.get("difficulty") == difficulty]
