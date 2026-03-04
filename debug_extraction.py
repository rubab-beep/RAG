# debug_extraction.py
# Save in project root and run: python debug_extraction.py

from utils.loader import load_documents
import sys

docs_dir = "./data/documents"   # or path to your uploaded files

pages = load_documents(docs_dir)

print(f"\nTotal pages extracted: {len(pages)}\n")
print("=" * 60)

for page in pages:
    print(f"\nFILE: {page['source']}  |  PAGE: {page['page']}")
    print(f"CHARACTERS: {len(page['text'])}")
    print(f"PREVIEW:")
    print(page['text'][:500])
    print("-" * 60)