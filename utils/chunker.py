"""
utils/chunker.py — Splits pages into overlapping chunks.
Metadata (source, page) is carried through every chunk.
Strategy: RecursiveCharacterTextSplitter respects paragraph → sentence → word order.
"""

from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from config import CHUNK_SIZE, CHUNK_OVERLAP, CHUNK_SEPARATORS


def chunk_pages(pages: List[Dict[str, Any]]) -> List[Document]:
    """
    Takes raw page dicts from loader.py and returns LangChain Document objects
    with chunked text and preserved metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=CHUNK_SEPARATORS,
        length_function=len,
    )

    all_chunks: List[Document] = []

    for page in pages:
        raw_text = page["text"]
        if not raw_text.strip():
            continue

        # Split this page's text into chunks
        text_chunks = splitter.split_text(raw_text)

        for idx, chunk_text in enumerate(text_chunks):
            # Each chunk carries full provenance — essential for citations
            doc = Document(
                page_content=chunk_text,
                metadata={
                    "source": page["source"],
                    "page": page["page"],
                    "chunk_index": idx,
                    "doc_type": page.get("doc_type", "unknown"),
                },
            )
            all_chunks.append(doc)

    print(f"  ✓ Created {len(all_chunks)} chunks from {len(pages)} pages")
    print(f"    Settings: chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")
    return all_chunks
