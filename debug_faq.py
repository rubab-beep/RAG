# debug_faq.py
import os
from dotenv import load_dotenv
load_dotenv()

from utils.embedder  import get_embedding_model
from utils.retriever import load_vectorstore

embedding_model = get_embedding_model()
vectorstore     = load_vectorstore(embedding_model)

question = "What are the frequently asked questions?"

# Get top 10 with NO threshold filter
results = vectorstore.similarity_search_with_relevance_scores(question, k=10)

print(f"Top 10 results — NO threshold applied\n")
print("=" * 60)

for i, (doc, score) in enumerate(results, 1):
    print(f"\nRank {i} | Score: {score:.4f} | Page: {doc.metadata.get('page')}")
    print(f"Preview: {doc.page_content[:150]}")
    print("-" * 40)