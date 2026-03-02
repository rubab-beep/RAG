"""
utils/loader.py — Document loading and text extraction.
Supports: PDF, DOCX, TXT.
Each page/section is returned as a dict with text + metadata.
"""

import os
from pathlib import Path
from typing import List, Dict, Any


def load_documents(docs_dir: str) -> List[Dict[str, Any]]:
    """
    Walk docs_dir and extract all text with source metadata.
    Returns a list of page-level dicts:
      { "text": str, "source": str, "page": int, "doc_type": str }
    """
    docs_dir = Path(docs_dir)
    all_pages = []

    supported = {".pdf": _load_pdf, ".PDF": _load_pdf, ".docx": _load_docx, ".DOCX": _load_docx, ".txt": _load_txt, ".TXT": _load_txt}

    found = list(docs_dir.glob("**/*"))
    files = [f for f in found if f.suffix.lower() in supported and f.is_file()]

    if not files:
        raise FileNotFoundError(
            f"No supported documents found in '{docs_dir}'.\n"
            f"Supported formats: {list(supported.keys())}\n"
            f"Drop your files there and re-run ingest.py."
        )

    for filepath in files:
        ext = filepath.suffix.lower()
        loader_fn = supported[ext]
        print(f"  Loading: {filepath.name}")
        try:
            pages = loader_fn(filepath)
            all_pages.extend(pages)
        except Exception as e:
            print(f"  ⚠️  Skipped {filepath.name}: {e}")

    print(f"\n  ✓ Loaded {len(files)} files → {len(all_pages)} pages extracted")
    return all_pages


# ── PDF ───────────────────────────────────────────────────────────────────────
def _load_pdf(filepath: Path) -> List[Dict[str, Any]]:
    import pdfplumber

    pages = []
    with pdfplumber.open(str(filepath)) as pdf:
        for page_num, page in enumerate(pdf.pages):

            # Extract regular text
            text = page.extract_text() or ""

            # Extract tables and convert to readable text
            tables = page.extract_tables()
            table_text = ""
            for table in tables:
                for row in table:
                    # Filter None cells, join with pipe separator
                    clean_row = [str(cell) if cell else "" for cell in row]
                    table_text += " | ".join(clean_row) + "\n"

            # Combine text and table content
            full_text = text + "\n" + table_text if table_text else text

            if len(full_text.strip()) < 20:
                continue

            pages.append({
                "text": full_text.strip(),
                "source": filepath.name,
                "page": page_num + 1,
                "doc_type": "pdf",
            })

    return pages



# ── DOCX ──────────────────────────────────────────────────────────────────────
def _load_docx(filepath: Path) -> List[Dict[str, Any]]:
    try:
        import docx
    except ImportError:
        raise ImportError("Run: pip install python-docx")

    doc = docx.Document(str(filepath))
    full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    # DOCX has no native pages — treat whole doc as page 1
    return [{
        "text": full_text,
        "source": filepath.name,
        "page": 1,
        "doc_type": "docx",
    }]


# ── TXT ───────────────────────────────────────────────────────────────────────
def _load_txt(filepath: Path) -> List[Dict[str, Any]]:
    text = filepath.read_text(encoding="utf-8", errors="replace").strip()
    return [{
        "text": text,
        "source": filepath.name,
        "page": 1,
        "doc_type": "txt",
    }]
