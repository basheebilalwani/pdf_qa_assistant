"""
text_splitter.py — Overlapping chunk generation with metadata propagation.

Provides one public function:
  - split_documents: splits a list of page Documents into overlapping chunks,
    propagating the 'page' metadata field from each source document to every
    derived chunk.
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """
    Split a list of page Documents into overlapping chunks.

    Each output chunk carries the same metadata (including 'page') as its
    source page Document.

    Returns [] if documents is empty or all pages have empty text.
    Does NOT raise exceptions for empty input.

    Valid ranges:
      - chunk_size: 100–2000 characters
      - chunk_overlap: 0 to chunk_size // 2
    """
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks: list[Document] = []
    for doc in documents:
        # Skip pages with no text — they would yield zero chunks anyway,
        # and RecursiveCharacterTextSplitter handles empty strings gracefully.
        if not doc.page_content:
            continue
        page_chunks = splitter.split_documents([doc])
        chunks.extend(page_chunks)

    return chunks
