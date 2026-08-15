"""
vector_store.py — ChromaDB collection lifecycle and retriever factory.

Provides three public functions and one constant:
  - CHROMA_PERSIST_DIR: local path where ChromaDB stores collections on disk
  - derive_collection_name: produce a deterministic, ChromaDB-safe name from a filename
  - build_vector_store: create (or replace) a ChromaDB collection and ingest chunks
  - get_retriever: load an existing collection and return a LangChain retriever

Design notes:
  - One collection per PDF; a re-upload replaces the old collection entirely.
  - Collection names are derived deterministically so the same file always maps
    to the same collection, enabling idempotent ingestion across restarts.
  - Embeddings always use the shared HuggingFaceEmbeddings instance passed by
    the caller — never instantiated inside this module.
"""

import re

import chromadb
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_huggingface import HuggingFaceEmbeddings

# ChromaDB local persistence directory (relative to project root)
CHROMA_PERSIST_DIR = "./chroma_db"

# Valid range for the retrieval top-k parameter
_K_MIN = 1
_K_MAX = 20


def derive_collection_name(filename: str) -> str:
    """
    Produce a deterministic, ChromaDB-safe collection name from a filename.

    Strategy:
      1. Strip the .pdf extension (case-insensitive) to get the stem.
      2. Lowercase the stem.
      3. Replace any character not in [a-z0-9_] with '_'.
      4. Prefix with 'pdf_' to guarantee the name starts with a letter
         and is always ≥ 4 chars.
      5. Strip trailing underscores/hyphens so the name ends with alphanumeric.
      6. Truncate to 63 characters.
      7. If the result is still < 3 chars (degenerate filename), pad with '_x'.

    Same filename always produces the same name (pure function, no side effects).

    Example:
      "My Report (2024).pdf" -> "pdf_my_report__2024_"  (trailing _ stripped)
                              -> "pdf_my_report__2024"
    """
    # Step 1: strip .pdf extension (case-insensitive)
    stem = re.sub(r"\.pdf$", "", filename, flags=re.IGNORECASE)

    # Step 2: lowercase
    stem = stem.lower()

    # Step 3: replace non-alphanumeric/underscore chars with '_'
    stem = re.sub(r"[^a-z0-9_]", "_", stem)

    # Step 4: prefix with 'pdf_'
    name = "pdf_" + stem

    # Step 5: strip trailing underscores and hyphens
    name = name.rstrip("_-")

    # Step 6: truncate to 63 characters (ChromaDB max)
    name = name[:63]

    # Step 7: pad if still too short (< 3 chars, degenerate filename)
    if len(name) < 3:
        name = name + "_x"

    return name


def build_vector_store(
    chunks: list[Document],
    collection_name: str,
    embedder: HuggingFaceEmbeddings,
) -> Chroma:
    """
    Create (or replace) a ChromaDB collection and ingest chunks.

    If a collection named collection_name already exists in CHROMA_PERSIST_DIR,
    it is deleted before recreation so that re-uploads don't accumulate stale
    chunks.

    Raises RuntimeError if deletion or creation fails.
    Returns the Chroma instance for the new collection.
    """
    # Connect to the persistent ChromaDB client
    try:
        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to connect to ChromaDB at '{CHROMA_PERSIST_DIR}': {exc}"
        ) from exc

    # Delete the existing collection if present
    try:
        client.delete_collection(collection_name)
    except (ValueError, chromadb.errors.NotFoundError):
        # Collection did not exist — nothing to delete, continue normally
        pass
    except Exception as exc:
        raise RuntimeError(
            f"Failed to delete existing collection '{collection_name}': {exc}"
        ) from exc

    # Create the collection and ingest all chunks
    try:
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embedder,
            collection_name=collection_name,
            persist_directory=CHROMA_PERSIST_DIR,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to create ChromaDB collection '{collection_name}': {exc}"
        ) from exc

    return vector_store


def get_retriever(
    collection_name: str,
    embedder: HuggingFaceEmbeddings,
    k: int = 4,
) -> BaseRetriever:
    """
    Load an existing ChromaDB collection and return a LangChain retriever.

    k: number of chunks to retrieve per query.
       Valid range: 1–20 (inclusive).

    Raises ValueError for k outside the valid range.
    Raises RuntimeError if the collection cannot be loaded.
    """
    if not (_K_MIN <= k <= _K_MAX):
        raise ValueError(
            f"k must be between {_K_MIN} and {_K_MAX} inclusive, got {k}."
        )

    try:
        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embedder,
            persist_directory=CHROMA_PERSIST_DIR,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load ChromaDB collection '{collection_name}': {exc}"
        ) from exc

    return vector_store.as_retriever(search_kwargs={"k": k})
