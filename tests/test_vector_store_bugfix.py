"""
tests/test_vector_store_bugfix.py

Bugfix validation suite for vector_store.py — ChromaDB collection lifecycle fix.

Task 1  — Bug condition exploration test (expected to FAIL on unfixed code).
Task 2  — Preservation property tests (expected to PASS on both unfixed and fixed code).
Tasks 3.2/3.3 — Same tests re-run after the fix; all must PASS.

Testing framework: pytest + hypothesis (already in requirements.txt).
Mocking: unittest.mock (stdlib).
"""

import pytest
from unittest.mock import MagicMock, patch

import chromadb
import chromadb.errors

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_community.vectorstores import Chroma

from vector_store import build_vector_store, get_retriever, derive_collection_name

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

SAMPLE_CHUNKS = [
    Document(page_content="Hello world", metadata={"page": 1}),
    Document(page_content="Foo bar baz", metadata={"page": 2}),
]
COLLECTION_NAME = "pdf_report"


def _make_mock_embedder():
    """Return a minimal mock that satisfies HuggingFaceEmbeddings duck-typing."""
    embedder = MagicMock()
    embedder.embed_documents = MagicMock(return_value=[[0.1, 0.2]])
    embedder.embed_query = MagicMock(return_value=[0.1, 0.2])
    return embedder


# ---------------------------------------------------------------------------
# Task 1 — Bug condition exploration test
#
# Property 1: First-time upload must NOT crash.
# On UNFIXED code this test FAILS with RuntimeError — that failure *is* the
# proof the bug exists.  After the fix (task 3.1) this test must PASS.
# Validates: Requirements 1.1, 1.2
# ---------------------------------------------------------------------------


def test_bug_condition_not_found_error_is_not_value_error():
    """
    Document the type-hierarchy root cause.

    chromadb.errors.NotFoundError must NOT be a subclass of ValueError.
    This assertion explains *why* the except ValueError guard fails to catch it.
    """
    assert not issubclass(chromadb.errors.NotFoundError, ValueError), (
        "NotFoundError IS a subclass of ValueError — the bug would not exist. "
        "Check the installed chromadb version."
    )


def test_bug_condition_first_time_upload_does_not_raise():
    """
    Property 1: Bug Condition — First-Time Upload Completes Without Exception.

    Validates: Requirements 2.1, 2.3

    When delete_collection raises chromadb.errors.NotFoundError (no collection
    exists yet), build_vector_store() must NOT raise any exception and must
    return a Chroma instance.

    ON UNFIXED CODE: this test FAILS with
        RuntimeError("Failed to delete existing collection …")
    That failure is the expected outcome for task 1 — it confirms the bug exists.

    ON FIXED CODE (after task 3.1): this test PASSES.
    """
    mock_chroma_instance = MagicMock(spec=Chroma)
    mock_client = MagicMock()
    mock_client.delete_collection.side_effect = chromadb.errors.NotFoundError(
        "Collection pdf_report does not exist."
    )

    with patch("vector_store.chromadb.PersistentClient", return_value=mock_client), \
         patch("vector_store.Chroma.from_documents", return_value=mock_chroma_instance):
        # On unfixed code this raises RuntimeError — confirming the bug.
        # On fixed code this must complete silently.
        result = build_vector_store(SAMPLE_CHUNKS, COLLECTION_NAME, _make_mock_embedder())

    assert isinstance(result, MagicMock), (
        "build_vector_store() should return the Chroma instance from from_documents"
    )


# ---------------------------------------------------------------------------
# Task 2 — Preservation property tests
#
# Property 2: Existing-Collection and Error-Propagation Behaviors Unchanged.
# All four subtests must PASS on both unfixed and fixed code.
# Validates: Requirements 3.1, 3.2, 3.3, 3.4
# ---------------------------------------------------------------------------


# --- 2a: Genuine-error preservation -----------------------------------------

@pytest.mark.parametrize("exc", [
    OSError("disk full"),
    PermissionError("permission denied"),
    RuntimeError("storage backend unavailable"),
    IOError("I/O error"),
])
def test_preservation_genuine_errors_raise_runtime_error(exc):
    """
    Property 2 — Genuine-error preservation.

    Validates: Requirement 3.2

    For any non-NotFoundError, non-ValueError exception raised by
    delete_collection(), build_vector_store() must re-raise it as RuntimeError.

    This property holds on both unfixed and fixed code — the fix only widens
    the no-op guard; it must NOT swallow genuine storage failures.
    """
    mock_client = MagicMock()
    mock_client.delete_collection.side_effect = exc

    with patch("vector_store.chromadb.PersistentClient", return_value=mock_client):
        with pytest.raises(RuntimeError) as exc_info:
            build_vector_store(SAMPLE_CHUNKS, COLLECTION_NAME, _make_mock_embedder())

    assert "Failed to delete existing collection" in str(exc_info.value), (
        "RuntimeError message should reference the collection deletion step"
    )


# --- 2b: Re-upload preservation ----------------------------------------------

def test_preservation_re_upload_returns_chroma_instance():
    """
    Property 2 — Re-upload preservation.

    Validates: Requirements 2.2, 3.1

    When delete_collection succeeds (existing-collection path), build_vector_store()
    must return a Chroma instance.  This path was never broken and must stay intact.
    """
    mock_chroma_instance = MagicMock(spec=Chroma)
    mock_client = MagicMock()
    mock_client.delete_collection.return_value = None  # succeeds silently

    with patch("vector_store.chromadb.PersistentClient", return_value=mock_client), \
         patch("vector_store.Chroma.from_documents", return_value=mock_chroma_instance):
        result = build_vector_store(SAMPLE_CHUNKS, COLLECTION_NAME, _make_mock_embedder())

    assert result is mock_chroma_instance, (
        "build_vector_store() must return the Chroma instance from from_documents"
    )


# --- 2c: get_retriever() unaffected ------------------------------------------

def test_preservation_get_retriever_returns_base_retriever():
    """
    Property 2 — get_retriever() unaffected.

    Validates: Requirement 3.3

    get_retriever() must return a BaseRetriever regardless of whether the fix
    has been applied.  The function is not modified by this bugfix.
    """
    mock_retriever = MagicMock(spec=BaseRetriever)
    mock_chroma_instance = MagicMock(spec=Chroma)
    mock_chroma_instance.as_retriever.return_value = mock_retriever

    with patch("vector_store.Chroma", return_value=mock_chroma_instance):
        result = get_retriever(COLLECTION_NAME, _make_mock_embedder(), k=4)

    assert isinstance(result, BaseRetriever), (
        "get_retriever() must return a BaseRetriever instance"
    )


# --- 2d: derive_collection_name() unaffected ---------------------------------

@pytest.mark.parametrize("filename,expected", [
    ("report.pdf",                "pdf_report"),
    ("My Report (2024).pdf",      "pdf_my_report__2024"),
    ("hello_world.PDF",           "pdf_hello_world"),
    # '-' is not in [a-z0-9_] so it gets replaced with '_'
    ("Annual-Report_2023.pdf",    "pdf_annual_report_2023"),
    ("  spaces  .pdf",            "pdf___spaces"),
    ("UPPERCASE.PDF",             "pdf_uppercase"),
    ("123numbers.pdf",            "pdf_123numbers"),
    ("a.pdf",                     "pdf_a"),
    # Chinese char → replaced by '_', stem becomes '_', trailing '_' stripped → 'pdf' (3 chars, no pad needed)
    ("已.pdf",                    "pdf"),
])
def test_preservation_derive_collection_name_deterministic(filename, expected):
    """
    Property 2 — derive_collection_name() unaffected.

    Validates: Requirement 3.4

    The function is pure and must return identical deterministic output before
    and after the fix.  These values are recorded observations on the unfixed
    code and must remain stable.
    """
    result = derive_collection_name(filename)
    assert result == expected, (
        f"derive_collection_name({filename!r}) returned {result!r}, expected {expected!r}"
    )
