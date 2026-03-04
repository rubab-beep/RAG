 
with open('utils/loader.py', 'a') as f:
    f.write("""

def _extract_paper_metadata(pages):
    import re
    if not pages:
        return pages
    first_page_text = pages[0]["text"]
    source = pages[0]["source"]
    emails = re.findall(r'[\\w\\.-]+@[\\w\\.-]+\\.\\w+', first_page_text)
    lines = first_page_text.split('\\n')
    name_pattern = re.compile(r'^[A-Z][a-z]+ [A-Z][a-z]+$')
    potential_names = [
        l.strip() for l in lines
        if name_pattern.match(l.strip()) and len(l.strip().split()) <= 4
    ]
    if not potential_names and not emails:
        return pages
    author_chunk_text = (
        "Paper authors and metadata information:\\n"
        "The authors of this paper are: " + ", ".join(potential_names) + ".\\n"
        "Author emails: " + ", ".join(emails) + ".\\n\\n"
        "Original page 1 content:\\n" + first_page_text[:800]
    )
    pages[0] = {"text": author_chunk_text, "source": source, "page": 1, "doc_type": "metadata"}
    return pages
""")
print("Done")