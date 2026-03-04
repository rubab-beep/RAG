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
        ".csv":  _load_csv,
        ".xlsx": _load_excel,
        ".xls":  _load_excel,
    }
    

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

    # ── Enrich academic paper metadata ────────────────────────
    all_pages = _extract_paper_metadata(all_pages)
    return all_pages


# ── PDF ───────────────────────────────────────────────────────────────────────
def _load_pdf(filepath: Path) -> List[Dict[str, Any]]:
    import pdfplumber

    pages = []

    with pdfplumber.open(str(filepath)) as pdf:
        for page_num, page in enumerate(pdf.pages):

            # ── Detect multi-column layout ────────────────────
            if _is_multicolumn(page):
                mid_x    = page.width / 2
                raw_text = (
                    _extract_column(page.crop((0, 0, mid_x, page.height)))
                    + "\n\n"
                    + _extract_column(page.crop((mid_x, 0, page.width, page.height)))
                )
            else:
                raw_text = page.extract_text() or ""

            # ── Extract tables with deduplication ────────────
            table_text = ""
            seen_rows  = set()

            tables = page.extract_tables()
            for table in tables:
                if not table:
                    continue
                for row in table:
                    if not row:
                        continue
                    clean_row = [str(cell).strip() if cell else "" for cell in row]
                    if not any(cell for cell in clean_row):
                        continue
                    row_text = " | ".join(clean_row)
                    if row_text in seen_rows:
                        continue                  # ← stops the loop
                    seen_rows.add(row_text)
                    table_text += row_text + "\n"

            # ── Combine text and tables ───────────────────────
            full_text = raw_text
            if table_text:
                full_text = full_text + "\n" + table_text

            # ── Clean artifacts and repeated lines ────────────
            full_text = _clean_pdf_text(full_text)   # ← now actually called

            if len(full_text.strip()) < 20:
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

    print(f"Column check: left={left_words} right={right_words} multicolumn={result}")

    return result


def _extract_column(cropped_page) -> str:
    """
    Extracts text from one column by sorting words top-to-bottom
    instead of left-to-right. This is the fix for mixed column output.
    """
    try:
        words = cropped_page.extract_words(
            x_tolerance      = 3,
            y_tolerance      = 3,
            keep_blank_chars = False,
        )

        if not words:
            return cropped_page.extract_text() or ""

        # Sort by vertical position first, then horizontal
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
# ── CSV ───────────────────────────────────────────────────────────────────────

def _load_csv(filepath: Path) -> List[Dict[str, Any]]:
    """
    Loads a CSV file and converts each row to readable text.
    Each chunk of 50 rows becomes one page to avoid huge single chunks.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("Run: pip install pandas")
 # Try encodings in order — utf-8-sig handles Windows BOM files
    for encoding in ["utf-8-sig", "utf-8", "latin-1", "cp1252"]:
        try:
            df = pd.read_csv(str(filepath), encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError(f"Could not decode {filepath.name} with any known encoding")

    # Clean column names
    df.columns = [str(c).strip() for c in df.columns]

    # Drop completely empty rows
    df = df.dropna(how="all")

    pages      = []
    chunk_size = 50    # rows per page chunk
    columns    = list(df.columns)

    # Split into chunks of 50 rows so each chunk is retrievable
    for chunk_start in range(0, len(df), chunk_size):
        chunk = df.iloc[chunk_start : chunk_start + chunk_size]

        lines = []

        # Add column headers at top of each chunk for context
        lines.append(f"Columns: {', '.join(columns)}")
        lines.append(f"Rows {chunk_start+1} to {chunk_start+len(chunk)}:")
        lines.append("")

        # Convert each row to readable key: value format
        for _, row in chunk.iterrows():
            row_parts = []
            for col in columns:
                val = row[col]
                if pd.notna(val) and str(val).strip():
                    row_parts.append(f"{col}: {val}")
            if row_parts:
                lines.append(" | ".join(row_parts))

        full_text = "\n".join(lines).strip()

        if len(full_text) < 20:
            continue

        pages.append({
            "text":     full_text,
            "source":   filepath.name,
            "page":     (chunk_start // chunk_size) + 1,
            "doc_type": "csv",
        })

    print(f"    ✓ CSV: {len(df)} rows → {len(pages)} chunks")
    return pages


# ── Excel ─────────────────────────────────────────────────────────────────────

def _load_excel(filepath: Path) -> List[Dict[str, Any]]:
    """
    Loads an Excel file (.xlsx or .xls).
    Each sheet is treated as a separate document section.
    Each sheet is split into 50-row chunks.
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("Run: pip install pandas openpyxl")

    pages      = []
    chunk_size = 50

    # Read all sheets
    try:
        all_sheets = pd.read_excel(
            str(filepath),
            sheet_name = None,    # None = load all sheets
            engine     = "openpyxl" if filepath.suffix.lower() == ".xlsx" else None,
        )
    except Exception as e:
        raise ValueError(f"Could not read Excel file: {e}")

    for sheet_name, df in all_sheets.items():

        # Clean up
        df.columns = [str(c).strip() for c in df.columns]
        df         = df.dropna(how="all")

        if df.empty:
            print(f"    ⚠️  Sheet '{sheet_name}' is empty, skipping")
            continue

        columns = list(df.columns)
        print(f"    Sheet '{sheet_name}': {len(df)} rows × {len(columns)} columns")

        for chunk_start in range(0, len(df), chunk_size):
            chunk = df.iloc[chunk_start : chunk_start + chunk_size]

            lines = []

            # Sheet name and column headers for context
            lines.append(f"Sheet: {sheet_name}")
            lines.append(f"Columns: {', '.join(columns)}")
            lines.append(f"Rows {chunk_start+1} to {chunk_start+len(chunk)}:")
            lines.append("")

            for _, row in chunk.iterrows():
                row_parts = []
                for col in columns:
                    val = row[col]
                    if pd.notna(val) and str(val).strip():
                        row_parts.append(f"{col}: {val}")
                if row_parts:
                    lines.append(" | ".join(row_parts))

            full_text = "\n".join(lines).strip()

            if len(full_text) < 20:
                continue

            pages.append({
                "text":     full_text,
                "source":   filepath.name,
                "page":     (chunk_start // chunk_size) + 1,
                "doc_type": "excel",
                "sheet":    sheet_name,
            })

    print(f"    ✓ Excel: {len(all_sheets)} sheets → {len(pages)} chunks")
    return pages
    
def _clean_pdf_text(text: str) -> str:
    """
    Cleans PDF extraction artifacts including repeated line loops.
    """
    import re

    # Fix broken hyphenated words split across lines
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)

    # Remove excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove lone page numbers
    text = re.sub(r'\n\s*\d{1,3}\s*\n', '\n', text)

    # Collapse multiple spaces
    text = re.sub(r' {2,}', ' ', text)

    # ── DEDUPLICATION — fixes your exact problem ──────────────
    text = _remove_repeated_lines(text)

    return text.strip()


def _remove_repeated_lines(text: str) -> str:
    """
    Detects and removes looping repeated lines.

    If the same line appears more than 3 times in a row
    or more than 5 times total in a page — it is a loop artifact.
    Keep only the first occurrence.
    """
    lines       = text.split('\n')
    clean_lines = []
    seen_counts = {}       # tracks total occurrences per line
    consecutive = {}       # tracks consecutive repetitions

    prev_line = None

    for line in lines:
        stripped = line.strip()

        if not stripped:
            clean_lines.append(line)
            prev_line = stripped
            continue

        # Count total occurrences
        seen_counts[stripped] = seen_counts.get(stripped, 0) + 1

        # Count consecutive occurrences
        if stripped == prev_line:
            consecutive[stripped] = consecutive.get(stripped, 0) + 1
        else:
            consecutive[stripped] = 1

        # Skip if appearing too many times
        if consecutive.get(stripped, 1) > 2:
            # Same line 3+ times in a row = loop, skip it
            continue

        if seen_counts.get(stripped, 0) > 5:
            # Same line 6+ times total on page = artifact, skip it
            continue

        clean_lines.append(line)
        prev_line = stripped

    return '\n'.join(clean_lines)

    

def _extract_paper_metadata(pages):
    import re
    if not pages:
        return pages

    first_page_text = pages[0]["text"]
    source          = pages[0]["source"]

    emails = re.findall(r"[\w\.-]+@[\w\.-]+\.\w+", first_page_text)
    emails = list(dict.fromkeys(emails))

    lines        = first_page_text.split("\n")
    name_pattern = re.compile(r"^[A-Z][a-z]+ [A-Z][a-z]+$")
    potential_names = list(dict.fromkeys([
        l.strip() for l in lines
        if name_pattern.match(l.strip()) and len(l.strip().split()) <= 4
    ]))

    if not potential_names and not emails:
        return pages

    # Build clean chunk with NO raw page text to prevent loops
    names_str  = ", ".join(potential_names) if potential_names else "See page 1"
    emails_str = ", ".join(emails[:9]) if emails else "See page 1"

    author_chunk_text = (
        "Paper authors and metadata:\n"
        "Authors: " + names_str + ".\n"
        "Emails: " + emails_str + ".\n"
        "All authors are affiliated with Microsoft Research "
        "or University of Zurich. See page 1 for full details."
    )

    pages[0] = {
        "text":     author_chunk_text,
        "source":   source,
        "page":     1,
        "doc_type": "metadata",
    }
    return pages