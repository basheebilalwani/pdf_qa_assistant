"""
pdf_loader.py — PDF validation and page extraction.

Provides two public functions:
  - validate_upload: checks filename extension and file size before loading
  - load_pdf: parses PDF bytes and returns one Document per page with page metadata
"""

import io

import pypdf
from langchain_core.documents import Document

# 200 MB in bytes
_MAX_SIZE_BYTES = 200 * 1024 * 1024


def validate_upload(filename: str, size_bytes: int) -> tuple[bool, str]:
    """
    Validate a file before loading.

    Returns (True, "") if valid.
    Returns (False, error_message) if:
      - filename does not end with ".pdf" (case-insensitive)
      - size_bytes > 200 MB

    Does NOT raise exceptions.
    """
    if not filename.lower().endswith(".pdf"):
        return False, "Only PDF files are supported."

    if size_bytes > _MAX_SIZE_BYTES:
        return False, "File exceeds 200 MB maximum."

    return True, ""


def load_pdf(file_bytes: bytes) -> list[Document]:
    """
    Parse PDF bytes and return one Document per page.

    Each Document has:
      - page_content: str  (text of the page, empty string if no text)
      - metadata: {"page": int}  (1-based page number)

    Documents are ordered by ascending page number.
    Raises ValueError if the bytes cannot be parsed as a valid PDF,
    or if the PDF contains zero pages.
    """
    try:
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
    except Exception as exc:
        raise ValueError(f"Could not parse PDF: {exc}") from exc

    if len(reader.pages) == 0:
        raise ValueError("PDF contains zero extractable pages.")

    documents = [
        Document(
            page_content=page.extract_text() or "",
            metadata={"page": page_num},
        )
        for page_num, page in enumerate(reader.pages, start=1)
    ]

    return documents
