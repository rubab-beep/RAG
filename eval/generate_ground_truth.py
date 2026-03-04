# eval/generate_ground_truth.py

"""
Automatically generates ground truth questions from uploaded documents.
Called after ingestion — reads the same chunks already in the vectorstore
and asks the LLM to generate test questions from them.

No manual editing needed when documents change.
"""

import os
import json
import time
from pathlib import Path
from typing  import List, Dict, Any


def generate_ground_truth(
    chunks,           # the same chunks produced by chunker.py
    llm_client,       # the same LLM already loaded in RAGEngine
    questions_per_doc: int = 3,    # how many questions to generate per document
    save_to_file:      bool = True  # write to eval/generated_ground_truth.json
) -> List[Dict[str, Any]]:
    """
    Reads document chunks and generates test questions automatically.

    Strategy:
      - Groups chunks by source document
      - Picks representative chunks from each document
      - Asks LLM to generate questions + expected keywords from each chunk
      - Also generates out-of-scope refusal questions automatically
      - Returns a ground truth list in the same format as ground_truth.py
    """

    from langchain.schema import HumanMessage

    ground_truth = []
    question_id  = 1

    # ── Group chunks by source document ──────────────────────────────────
    docs = {}
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        if source not in docs:
            docs[source] = []
        docs[source].append(chunk)

    print(f"\n  Generating ground truth for {len(docs)} document(s)...")

    # ── Generate answerable questions for each document ───────────────────
    for source_file, doc_chunks in docs.items():

        print(f"    Processing: {source_file}")

        # Pick evenly spaced chunks to cover the whole document
        step        = max(1, len(doc_chunks) // questions_per_doc)
        sample_chunks = doc_chunks[::step][:questions_per_doc]

        for chunk in sample_chunks:
            prompt = f"""You are building a test dataset for an AI Q&A system.

Read this text excerpt from a document:

SOURCE FILE: {source_file}
PAGE: {chunk.metadata.get('page', 1)}
TEXT: {chunk.page_content}

Generate ONE good test question that:
1. Can be answered directly from this text
2. Is specific (not vague like "what is this about?")
3. Would require reading this exact section to answer

Then identify 3-5 keywords that MUST appear in a correct answer.

Respond ONLY in this exact JSON format, no other text:
{{
  "question": "your question here",
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "difficulty": "easy" or "medium" or "hard"
}}"""

            try:
                response = llm_client.invoke([HumanMessage(content=prompt)])
                raw      = response.content.strip()

                # Strip markdown code fences if LLM added them
                import re
                raw = re.sub(r"```json|```", "", raw).strip()

                parsed = json.loads(raw)

                ground_truth.append({
                    "id":              f"Q{question_id:02d}",
                    "question":        parsed["question"],
                    "relevant_source": Path(source_file).stem,  # filename without extension
                    "relevant_page":   chunk.metadata.get("page", 1),
                    "answer_keywords": parsed.get("keywords", []),
                    "should_refuse":   False,
                    "category":        Path(source_file).stem.lower().replace(" ", "_"),
                    "difficulty":      parsed.get("difficulty", "medium"),
                    "auto_generated":  True,
                })

                question_id += 1
                print(f"      ✓ Q{question_id-1:02d}: {parsed['question'][:60]}...")

                # Small delay to avoid hitting Groq rate limits
                time.sleep(0.5)

            except Exception as e:
                print(f"      ⚠️  Failed to generate question: {e}")
                continue

    # ── Add standard out-of-scope refusal questions ───────────────────────
    # These never change regardless of what documents you upload
    refusal_questions = [
        {
            "id":              f"R{i+1:02d}",
            "question":        q,
            "relevant_source": None,
            "relevant_page":   None,
            "answer_keywords": [],
            "should_refuse":   True,
            "category":        "out_of_scope",
            "difficulty":      "easy",
            "auto_generated":  True,
        }
        for i, q in enumerate([
            "What is the current stock price of Apple?",
            "Who won the last FIFA World Cup?",
            "What is the population of Pakistan?",
            "What will the weather be like tomorrow?",
            "Who is the current prime minister of UK?",
        ])
    ]

    ground_truth.extend(refusal_questions)

    # ── Save to JSON file ─────────────────────────────────────────────────
    if save_to_file:
        os.makedirs("./eval", exist_ok=True)
        output_path = "./eval/generated_ground_truth.json"

        with open(output_path, "w") as f:
            json.dump(ground_truth, f, indent=2)

        print(f"\n  ✓ Generated {len(ground_truth)} questions")
        print(f"  ✓ Saved to: {output_path}")

    return ground_truth


def load_generated_ground_truth() -> List[Dict[str, Any]]:
    """
    Loads the auto-generated ground truth from JSON.
    Returns empty list if not yet generated.
    """
    path = Path("./eval/generated_ground_truth.json")

    if not path.exists():
        return []

    with open(path) as f:
        return json.load(f)