# test_retrieval.py — save this in your project root
import os
from dotenv import load_dotenv
load_dotenv()

from utils.embedder import get_embedding_model
from utils.retriever import load_vectorstore, retrieve_chunks

# Load the engine components
print("Loading embedding model...")
embedding_model = get_embedding_model()
vectorstore = load_vectorstore(embedding_model)

# Test any question
question = "How to make good coffee?"

chunks, scores, top_score = retrieve_chunks(question, vectorstore)

print(f"\nQuestion: {question}")
print(f"Top score: {top_score:.3f}")
print(f"Chunks found: {len(chunks)}")
print("\n" + "="*60)

for i, (chunk, score) in enumerate(zip(chunks, scores), 1):
    print(f"\nCHUNK {i}")
    print(f"Score:  {score:.3f}")
    print(f"Source: {chunk.metadata['source']}")
    print(f"Page:   {chunk.metadata['page']}")
    print(f"Text:   {chunk.page_content[:200]}...")
    print("-"*60)