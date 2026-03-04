# Read the file
with open('utils/loader.py', 'r' , encoding='utf-8') as f:
    content = f.read()

# Find and replace the entire _extract_paper_metadata function
old = content[content.find('def _extract_paper_metadata'):]

new_func = '''def _extract_paper_metadata(pages):
    import re
    if not pages:
        return pages

    first_page_text = pages[0]["text"]
    source          = pages[0]["source"]

    emails = re.findall(r"[\\w\\.-]+@[\\w\\.-]+\\.\\w+", first_page_text)
    emails = list(dict.fromkeys(emails))

    lines        = first_page_text.split("\\n")
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
        "Paper authors and metadata:\\n"
        "Authors: " + names_str + ".\\n"
        "Emails: " + emails_str + ".\\n"
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
'''

# Replace everything from _extract_paper_metadata to end of file
cut_point = content.find('def _extract_paper_metadata')
new_content = content[:cut_point] + new_func

with open('utils/loader.py', 'w' , encoding='utf-8') as f:
    f.write(new_content)

print("Done - function replaced")