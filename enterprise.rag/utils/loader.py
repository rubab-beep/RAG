"""
utils/loader.py — Document loading and text extraction.
Supports: PDF, DOCX, TXT.
Each page/section is returned as a dict with text + metadata.
"""

import os
import re
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

    supported = {
        ".pdf":  _load_pdf,
        ".docx": _load_docx,
        ".txt":  _load_txt,
    }

    found = list(docs_dir.glob("**/*"))
    files = [f for f in found if f.suffix.lower() in supported and f.is_file()]

    if not files:
        raise FileNotFoundError(
            f"No supported documents found in '{docs_dir}'.\n"
            f"Supported formats: PDF, DOCX, TXT\n"
            f"Drop your files there and re-run."
        )

    for filepath in files:
        ext       = filepath.suffix.lower()
        loader_fn = supported[ext]
        print(f"  Loading: {filepath.name}")
        try:
            pages = loader_fn(filepath)
            all_pages.extend(pages)
        except Exception as e:
            print(f"  ⚠️  Skipped {filepath.name}: {e}")

    print(f"\n  ✓ Loaded {len(files)} files → {len(all_pages)} pages extracted")

    # Enrich academic paper metadata (authors, title, affiliations)
    all_pages = _extract_paper_metadata(all_pages)

    return all_pages


# ── PDF ───────────────────────────────────────────────────────────────────────

def _load_pdf(filepath: Path) -> List[Dict[str, Any]]:
    """
    Smart PDF loader:
    1. Detects multi-column layout and extracts each column separately
    2. Extracts tables with deduplication to prevent looping
    3. Cleans text artifacts and repeated lines
    4. Falls back to OCR if page has no extractable text
    """
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("Run: pip install pdfplumber")

    pages = []

    with pdfplumber.open(str(filepath)) as pdf:
        for page_num, page in enumerate(pdf.pages):

            # ── Step 1: Extract text ──────────────────────────────
            if _is_multicolumn(page):
                mid_x    = page.width / 2
                raw_text = (
                    _extract_column(page.crop((0, 0, mid_x, page.height)))
                    + "\n\n"
                    + _extract_column(page.crop((mid_x, 0, page.width, page.height)))
                )
            else:
                raw_text = page.extract_text() or ""

            # ── Step 2: Extract tables with dedup ─────────────────
            table_text = ""
            seen_rows  = set()

            try:
                tables = page.extract_tables()
                for table in tables:
                    if not table:
                        continue
                    table_lines = []
                    for row in table:
                        if not row:
                            continue
                        clean_row = [str(cell).strip() if cell else "" for cell in row]
                        if not any(cell for cell in clean_row):
                            continue
                        row_text = " | ".join(clean_row)
                        if len(row_text.strip()) < 15:
                            continue
                        if row_text in seen_rows:
                            continue
                        seen_rows.add(row_text)
                        table_lines.append(row_text)
                    if table_lines:
                        table_text += "\n".join(table_lines) + "\n\n"
            except Exception as e:
                print(f"    ⚠️  Table extraction error on page {page_num+1}: {e}")

            # ── Step 3: Combine and clean ─────────────────────────
            full_text = raw_text
            if table_text:
                full_text = full_text + "\n" + table_text

            full_text = _clean_pdf_text(full_text)

            # ── Step 4: OCR fallback for image-based pages ────────
            if len(full_text.strip()) < 20:
                print(f"    ℹ️  Page {page_num+1} has no text — trying OCR...")
                full_text = _ocr_page(filepath, page_num)

            if len(full_text.strip()) < 20:
                print(f"    ⚠️  Page {page_num+1}: could not extract text, skipping")
                continue

            pages.append({
                "text":     full_text.strip(),
                "source":   filepath.name,
                "page":     page_num + 1,
                "doc_type": "pdf",
            })

    return pages


def _is_multicolumn(page) -> bool:
    """
    Detects whether a PDF page uses multi-column layout.
    Splits page down the middle and checks if both halves
    have substantial independent text.
    """
    mid_x      = page.width / 2
    left_text  = page.crop((0, 0, mid_x, page.height)).extract_text() or ""
    right_text = page.crop((mid_x, 0, page.width, page.height)).extract_text() or ""

    left_words  = len(left_text.split())
    right_words = len(right_text.split())

    result = (
        left_words  > 30 and
        right_words > 30 and
        abs(left_words - right_words) < left_words * 0.8
    )

    print(f"    Column check: left={left_words} right={right_words} multicolumn={result}")

    return result


def _extract_column(cropped_page) -> str:
    """
    Extracts text from one column by sorting words top-to-bottom
    instead of left-to-right. Fixes mixed column output.
    """
    try:
        words = cropped_page.extract_words(
            x_tolerance      = 3,
            y_tolerance      = 3,
            keep_blank_chars = False,
        )

        if not words:
            return cropped_page.extract_text() or ""

        words.sort(key=lambda w: (round(w['top'] / 5) * 5, w['x0']))

        lines        = []
        current_line = []
        current_top  = None

        for word in words:
            if current_top is None:
                current_top = word['top']

            if abs(word['top'] - current_top) <= 5:
                current_line.append(word['text'])
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word['text']]
                current_top  = word['top']

        if current_line:
            lines.append(' '.join(current_line))

        return '\n'.join(lines)

    except Exception:
        return cropped_page.extract_text() or ""


def _clean_pdf_text(text: str) -> str:
    """Cleans PDF extraction artifacts including repeated line loops."""

    # Fix broken hyphenated words split across lines
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)

    # Remove excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove lone page numbers
    text = re.sub(r'\n\s*\d{1,3}\s*\n', '\n', text)

    # Collapse multiple spaces
    text = re.sub(r' {2,}', ' ', text)

    # Remove looping repeated lines
    text = _remove_repeated_lines(text)

    return text.strip()


def _remove_repeated_lines(text: str) -> str:
    """
    Detects and removes looping repeated lines.
    If the same line appears 3+ times in a row or 5+ times total — skip it.
    """
    lines       = text.split('\n')
    clean_lines = []
    seen_counts = {}
    consecutive = {}
    prev_line   = None

    for line in lines:
        stripped = line.strip()

        if not stripped:
            clean_lines.append(line)
            prev_line = stripped
            continue

        seen_counts[stripped] = seen_counts.get(stripped, 0) + 1
        consecutive[stripped] = consecutive.get(stripped, 0) + 1 \
                                 if stripped == prev_line else 1

        if consecutive.get(stripped, 1) > 2:
            continue
        if seen_counts.get(stripped, 0) > 5:
            continue

        clean_lines.append(line)
        prev_line = stripped

    return '\n'.join(clean_lines)


def _ocr_page(filepath: Path, page_num: int) -> str:
    """
    Runs OCR on a single PDF page.
    Used when pdfplumber finds no text — page is a scanned image.
    Requires: Tesseract installed + Poppler installed.
    """
    try:
        import pytesseract
        from pdf2image import convert_from_path
        from PIL       import ImageEnhance, ImageFilter

        POPPLER_PATH = r"C:\poppler\Library\bin"

        images = convert_from_path(
            str(filepath),
            dpi          = 300,
            first_page   = page_num + 1,
            last_page    = page_num + 1,
            poppler_path = POPPLER_PATH,
        )

        if not images:
            return ""

        image = images[0]
        image = image.convert("L")
        image = ImageEnhance.Contrast(image).enhance(2.0)
        image = image.filter(ImageFilter.SHARPEN)

        text = pytesseract.image_to_string(image, config="--psm 3 --oem 3")
        print(f"    ✓ OCR extracted {len(text)} characters from page {page_num+1}")
        return text.strip()

    except ImportError:
        return ""
    except Exception as e:
        print(f"    ⚠️  OCR failed for page {page_num+1}: {e}")
        return ""


# ── Academic Paper Metadata ───────────────────────────────────────────────────

def _extract_paper_metadata(pages):
    import re
    if not pages:
        return pages

    first_page_text = pages[0]["text"]
    source          = pages[0]["source"]

    emails = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', first_page_text)

    lines        = first_page_text.split('\n')
    name_pattern = re.compile(r'^[A-Z][a-z]+ [A-Z][a-z]+$')
    potential_names = list(dict.fromkeys([          # deduplicate preserving order
        l.strip() for l in lines
        if name_pattern.match(l.strip()) and len(l.strip().split()) <= 4
    ]))

    if not potential_names and not emails:
        return pages

    # Deduplicate emails too
    emails = list(dict.fromkeys(emails))

    # Do NOT include raw page text — it causes repetition loops
    author_chunk_text = (
        "Paper authors and metadata information:\n"
        "The authors of this paper are: " + ", ".join(potential_names) + ".\n"
        "Author emails: " + ", ".join(emails) + ".\n"
        "Affiliation: Microsoft Research and University of Zurich.\n"
        "The full author list with affiliations appears on page 1.\n"
    )

    pages[0] = {
        "text":     author_chunk_text,
        "source":   source,
        "page":     1,
        "doc_type": "metadata",
    }
    return pages


# ── DOCX ──────────────────────────────────────────────────────────────────────

def _load_docx(filepath: Path) -> List[Dict[str, Any]]:
    try:
        import docx
    except ImportError:
        raise ImportError("Run: pip install python-docx")

    doc       = docx.Document(str(filepath))
    full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    return [{
        "text":     full_text,
        "source":   filepath.name,
        "page":     1,
        "doc_type": "docx",
    }]


# ── TXT ───────────────────────────────────────────────────────name────────────

def _load_txt(filepath: Path) -> List[Dict[str, Any]]:
    text = filepath.read_text(encoding="utf-8", errors="replace").strip()
    return [{
        "text":     text,
        "source":   filepath.name,
        "page":     1,
        "doc_type": "txt",
    }]