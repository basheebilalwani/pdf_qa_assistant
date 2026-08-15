from langchain_core.documents import Document


def format_citations(source_docs: list[Document]) -> str:
    """
    Extract page numbers from source_docs metadata, deduplicate, sort
    ascending, and return a formatted string.

    Format: "Source pages: [1, 3, 5]"
    If source_docs is empty, returns "".
    If a doc is missing the 'page' metadata key, that doc's page is omitted;
    if some pages could not be determined, appends
    "(some source pages could not be determined)" to the output.
    """
    if not source_docs:
        return ""

    pages = []
    missing_page = False

    for doc in source_docs:
        if "page" in doc.metadata:
            pages.append(doc.metadata["page"])
        else:
            missing_page = True

    sorted_pages = sorted(set(pages))
    result = f"Source pages: {sorted_pages}"

    if missing_page:
        result += " (some source pages could not be determined)"

    return result
