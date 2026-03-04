from utils.loader import load_documents

pages = load_documents("./data/documents")

for page in pages:
    if "sample-unstructured-paper" in page["source"]:
        print(f"\nPage {page['page']} — {len(page['text'])} chars")
        print(page["text"][:400])
        print("-" * 50)