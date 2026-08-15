"""
app.py — Streamlit entry point for the PDF QA RAG system.

Startup order:
  1. Load configuration (module level) — display error and stop on EnvironmentError.
  2. Initialize Streamlit session state keys (only once per session).
  3. Cache shared objects (embedder, LLM) in session state so they are
     created exactly once per session regardless of Streamlit reruns.
  4. Render the UI (ingestion panel, query panel, conversation history).

Tasks 9.2–9.4 fill in sections marked TODO below.
"""

import streamlit as st

from config import load_config
from embeddings import get_embedder
from qa_chain import get_llm, run_qa_chain, FALLBACK_RESPONSE
from citations import format_citations
from pdf_loader import validate_upload, load_pdf
from text_splitter import split_documents
from vector_store import derive_collection_name, build_vector_store, get_retriever

# ---------------------------------------------------------------------------
# 1. Configuration loading — must happen at module level so that a missing
#    GROQ_API_KEY is caught before any UI is rendered (Requirements 7.8, 11.4).
# ---------------------------------------------------------------------------
try:
    _config = load_config()
except EnvironmentError as _env_err:
    st.error(
        f"Required environment variable is not set: **{_env_err}**\n\n"
        "Create a `.env` file in the project root (see `.env.example`) "
        "and set the missing variable, then restart the app."
    )
    st.stop()

# ---------------------------------------------------------------------------
# 2. Session state initialization — use setdefault so values are only set on
#    the very first run; subsequent reruns leave existing state untouched
#    (Requirements 11.1, 11.3).
# ---------------------------------------------------------------------------
st.session_state.setdefault("collection_name", None)   # str | None
st.session_state.setdefault("retriever", None)         # BaseRetriever | None
st.session_state.setdefault("llm", None)               # ChatGroq | None
st.session_state.setdefault("embedder", None)          # HuggingFaceEmbeddings | None
st.session_state.setdefault("history", [])             # list[ConversationEntry]
st.session_state.setdefault("uploaded_filename", None) # str | None

# ---------------------------------------------------------------------------
# 3. Shared object caching — create embedder and LLM once per session.
#    Subsequent reruns skip creation because the values are already set
#    (Requirements 7.6, 7.8, 11.1).
# ---------------------------------------------------------------------------
if st.session_state["embedder"] is None:
    with st.spinner("Loading embedding model…"):
        st.session_state["embedder"] = get_embedder()

if st.session_state["llm"] is None:
    st.session_state["llm"] = get_llm(_config["GROQ_API_KEY"])

# ---------------------------------------------------------------------------
# 4. Page layout
# ---------------------------------------------------------------------------
st.title("PDF QA Assistant")

# ---------------------------------------------------------------------------
# 4a. Ingestion pipeline UI (task 9.2)
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None and uploaded_file.name != st.session_state["uploaded_filename"]:
    with st.spinner("Processing PDF…"):
        # Step 1: validate
        valid, err_msg = validate_upload(uploaded_file.name, uploaded_file.size)
        if not valid:
            st.error(err_msg)
            st.stop()

        # Step 2: load pages
        try:
            pages = load_pdf(uploaded_file.read())
        except Exception as e:
            st.error(str(e))
            st.stop()

        # Step 3: split into chunks
        try:
            chunks = split_documents(pages)
        except Exception as e:
            st.error(str(e))
            st.stop()

        # Step 4: derive collection name
        collection_name = derive_collection_name(uploaded_file.name)

        # Step 5: build vector store
        try:
            build_vector_store(chunks, collection_name, st.session_state["embedder"])
        except RuntimeError as e:
            st.error(str(e))
            st.stop()

        # Step 6: get retriever
        try:
            retriever = get_retriever(collection_name, st.session_state["embedder"])
        except Exception as e:
            st.error(str(e))
            st.stop()

        # Commit to session state and clear history
        st.session_state["collection_name"] = collection_name
        st.session_state["retriever"] = retriever
        st.session_state["history"] = []
        st.session_state["uploaded_filename"] = uploaded_file.name

        st.success(f"Loaded: {uploaded_file.name}")

# Boolean flag used by tasks 9.3 and 9.4 to disable question input
pdf_loaded = st.session_state["retriever"] is not None

# Show current document status
if st.session_state["uploaded_filename"]:
    st.info(f"Active document: **{st.session_state['uploaded_filename']}**")
else:
    st.warning("Please upload a PDF to begin asking questions.")

# ---------------------------------------------------------------------------
# 4b. Conversation history display
#   - Renders session_state["history"] oldest (top) → newest (bottom)
#   - Each turn uses st.chat_message() for a ChatGPT-style layout
#   - Shows an empty-state message when no questions have been asked
#   - Omits citations when the answer is FALLBACK_RESPONSE (Requirement 9.3)
# ---------------------------------------------------------------------------
if not st.session_state["history"]:
    st.info("No conversation history yet. Upload a PDF and ask a question to get started.")
else:
    for entry in st.session_state["history"]:
        with st.chat_message("user"):
            st.markdown(entry["question"])
        with st.chat_message("assistant"):
            st.markdown(entry["answer"])
            if entry["answer"] != FALLBACK_RESPONSE and entry["citations"]:
                st.caption(entry["citations"])

# ---------------------------------------------------------------------------
# 4c. Chat input
#   - Replaces the st.text_input + st.button combo
#   - Disabled when no PDF is loaded
#   - Echoes the question immediately, then streams the answer in place
# ---------------------------------------------------------------------------
question = st.chat_input("Ask a question about the document…", disabled=not pdf_loaded)

if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching for an answer…"):
            try:
                answer, source_docs = run_qa_chain(
                    question,
                    st.session_state["retriever"],
                    st.session_state["llm"],
                )
            except RuntimeError as e:
                st.error(str(e))
            else:
                citations_str = format_citations(source_docs)
                st.session_state["history"].append(
                    {
                        "question": question,
                        "answer": answer,
                        "citations": citations_str,
                    }
                )
                st.markdown(answer)
                if answer != FALLBACK_RESPONSE and citations_str:
                    st.caption(citations_str)
