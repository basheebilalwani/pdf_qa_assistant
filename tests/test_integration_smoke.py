# Feature: pdf-qa-rag-system, Integration smoke test
"""
tests/test_integration_smoke.py

End-to-end integration smoke test for the PDF QA RAG pipeline.

Validates: Requirements 2.1, 3.2, 5.1, 6.2, 8.1, 9.2, 12.1

Pipeline under test:
  load_pdf → split_documents → build_vector_store → get_retriever
             → run_qa_chain (LLM mocked) → format_citations

The LLM (ChatGroq) is mocked so no real API calls are made.
ChromaDB storage is isolated to pytest's tmp_path; cleanup is automatic.
The embedder (HuggingFaceEmbeddings) runs locally — it is NOT mocked.
"""

import io
from unittest.mock import MagicMock, patch

import pytest
import pypdf
from pypdf import PdfWriter
from langchain_core.messages import AIMessage

import vector_store as vs
from pdf_loader import load_pdf
from text_splitter import split_documents
from embeddings import get_embedder
from vector_store import build_vector_store, get_retriever, derive_collection_name
from qa_chain import run_qa_chain
from citations import format_citations


# ---------------------------------------------------------------------------
# Helper: build a small synthetic PDF in memory
# ---------------------------------------------------------------------------

def _make_synthetic_pdf_bytes(pages: list[str]) -> bytes:
    """
    Write a multi-page PDF to a bytes buffer using pypdf.PdfWriter.
    Each entry in `pages` becomes one page of the PDF.
    """
    writer = PdfWriter()
    for text in pages:
        page = writer.add_blank_page(width=612, height=792)
        page.merge_page(page)  # ensure page object exists
        # Annotate the page with actual text via a simple content stream
        # pypdf.PdfWriter supports adding text via compress_content_streams,
        # but the simplest portable approach is to write raw PDF content.
        # We embed text using a minimal content stream appended to the page.
        _add_text_to_page(writer, page, text)

    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


def _add_text_to_page(writer: PdfWriter, page, text: str) -> None:
    """
    Embed `text` into a pypdf page via a raw content stream so that
    pypdf.PdfReader.extract_text() can read it back.
    """
    from pypdf.generic import (
        ArrayObject,
        ContentStream,
        DecodedStreamObject,
        NameObject,
        RectangleObject,
    )

    # Build a minimal PDF content stream: select font, set position, show text
    safe_text = text.replace("(", r"\(").replace(")", r"\)").replace("\\", "\\\\")
    stream_data = (
        "BT\n"
        "/F1 12 Tf\n"
        "72 720 Td\n"
        f"({safe_text}) Tj\n"
        "ET\n"
    ).encode("latin-1")

    stream_obj = DecodedStreamObject()
    stream_obj.set_data(stream_data)
    stream_ref = writer._add_object(stream_obj)

    # Attach font resource so the PDF is well-formed (best-effort)
    if "/Resources" not in page:
        page[NameObject("/Resources")] = writer._add_object(
            pypdf.generic.DictionaryObject()
        )

    resources = page["/Resources"]
    if isinstance(resources, pypdf.generic.IndirectObject):
        resources = resources.get_object()

    font_dict = pypdf.generic.DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    font_ref = writer._add_object(font_dict)

    if NameObject("/Font") not in resources:
        resources[NameObject("/Font")] = pypdf.generic.DictionaryObject()

    font_resources = resources[NameObject("/Font")]
    if isinstance(font_resources, pypdf.generic.IndirectObject):
        font_resources = font_resources.get_object()
    font_resources[NameObject("/F1")] = font_ref

    # Append the content stream to the page
    if "/Contents" in page:
        existing = page["/Contents"]
        if isinstance(existing, pypdf.generic.IndirectObject):
            page[NameObject("/Contents")] = ArrayObject(
                [existing, stream_ref]
            )
        elif isinstance(existing, ArrayObject):
            existing.append(stream_ref)
    else:
        page[NameObject("/Contents")] = stream_ref


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def test_end_to_end_smoke(tmp_path):
    """
    Full pipeline smoke test: load → split → embed → store → retrieve → QA → cite.

    - Synthetic PDF with 2 pages of meaningful text.
    - LLM is mocked; no Groq API calls are made.
    - ChromaDB writes to tmp_path (isolated, auto-cleaned by pytest).
    - Asserts:
        * answer is a str
        * citations starts with "Source pages: " or is ""
    """
    # ---- 1. Build a synthetic 2-page PDF in memory -------------------------
    page_texts = [
        (
            "Introduction to Machine Learning. "
            "Machine learning is a branch of artificial intelligence. "
            "It enables computers to learn from data without being explicitly programmed."
        ),
        (
            "Supervised Learning. "
            "In supervised learning, the model is trained on labelled examples. "
            "Common algorithms include linear regression, decision trees, and neural networks."
        ),
    ]
    pdf_bytes = _make_synthetic_pdf_bytes(page_texts)

    # ---- 2. Load PDF --------------------------------------------------------
    documents = load_pdf(pdf_bytes)
    assert len(documents) == 2, f"Expected 2 pages, got {len(documents)}"

    # ---- 3. Split into chunks -----------------------------------------------
    chunks = split_documents(documents, chunk_size=500, chunk_overlap=50)
    assert len(chunks) >= 1, "split_documents should return at least 1 chunk"

    # ---- 4. Get embedder (real HuggingFace model, runs locally) -------------
    embedder = get_embedder()

    # ---- 5. Build vector store with ChromaDB isolated to tmp_path -----------
    collection_name = derive_collection_name("smoke_test.pdf")

    with patch.object(vs, "CHROMA_PERSIST_DIR", str(tmp_path)):
        vector_store_instance = build_vector_store(chunks, collection_name, embedder)

        # ---- 6. Get retriever -----------------------------------------------
        retriever = get_retriever(collection_name, embedder, k=2)

        # ---- 7. Mock the LLM ------------------------------------------------
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = AIMessage(
            content="Machine learning enables computers to learn from data."
        )

        # ---- 8. Run QA chain ------------------------------------------------
        question = "What is machine learning?"
        answer, source_docs = run_qa_chain(question, retriever, mock_llm)

        # ---- 9. Format citations --------------------------------------------
        citations = format_citations(source_docs)

    # ---- 10. Assertions -----------------------------------------------------
    assert isinstance(answer, str), (
        f"answer must be a str, got {type(answer).__name__!r}"
    )
    assert citations == "" or citations.startswith("Source pages: "), (
        f"citations must be '' or start with 'Source pages: ', got {citations!r}"
    )
