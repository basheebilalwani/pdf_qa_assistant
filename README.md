# PDF QA Assistant

A local Retrieval-Augmented Generation (RAG) application that lets you upload any PDF and ask natural language questions about its content. Answers are grounded exclusively in the document — the app never invents information. If the answer isn't in the PDF, it says so.

---

## Demo

> Live demo: https://pdfappassistant.streamlit.app/

---

## Features

- **Upload any PDF** up to 200 MB via a drag-and-drop file uploader
- **Page-aware extraction** — every chunk retains the page number it came from
- **Overlapping chunking** — text is split with configurable overlap to preserve context across boundaries
- **Local embeddings** — `sentence-transformers/all-MiniLM-L6-v2` runs entirely on your machine; no embedding API is called
- **Isolated vector collections** — each PDF gets its own ChromaDB collection; chunks from different documents are never mixed
- **Re-upload without stale data** — uploading a new version of the same file replaces the old collection atomically
- **Groq LLM answers** — uses `llama-3.1-8b-instant` via the Groq API for fast, free-tier inference
- **Grounded answers only** — the prompt explicitly instructs the LLM to use only the retrieved context
- **Hallucination fallback** — when retrieved chunks contain no relevant information, the app returns `"Not found in the document."` and suppresses citations
- **Page citations** — every answer is followed by the source page numbers (e.g. `Source pages: [2, 5]`)
- **Conversation history** — the full Q&A session is preserved in a chat-style view for the duration of your session
- **Zero paid dependencies** — HuggingFace embeddings are free and local; Groq has a generous free tier

---

## Architecture / Pipeline

```
PDF upload
  │
  ▼
page-aware text extraction      pdf_loader.py
  │  (one Document per page, metadata: {page: N})
  ▼
overlapping chunking             text_splitter.py
  │  (RecursiveCharacterTextSplitter, 1000 chars / 200 overlap)
  ▼
HuggingFace embeddings           embeddings.py
  │  (sentence-transformers/all-MiniLM-L6-v2, runs locally)
  ▼
ChromaDB (local)                 vector_store.py
  │  (persistent, one collection per PDF)
  ▼
similarity retrieval             vector_store.py → get_retriever
  │  (top-k=4 chunks by cosine similarity)
  ▼
Groq LLM (llama-3.1-8b-instant)  qa_chain.py
  │  (context-only prompt; fallback if no relevant chunks found)
  ▼
grounded answer + page citations  citations.py
```

---

## Tech Stack

| Layer | Library / Service | Version |
|---|---|---|
| UI | Streamlit | 1.40.2 |
| LLM | Groq (`langchain-groq`) | 1.1.3 |
| Embeddings | `langchain-huggingface` + sentence-transformers | 1.2.2 / 5.7.0 |
| Vector DB | ChromaDB | 1.5.9 |
| LangChain core | `langchain`, `langchain-core`, `langchain-community` | 1.3.14 / 1.5.5 / 0.4.2 |
| PDF parsing | pypdf | 6.16.1 |
| Config | python-dotenv | 1.1.0 |
| Testing | pytest + hypothesis | 8.3.5 / 6.135.2 |

---

## How RAG Works in This Project

Standard LLMs have no knowledge of your private documents. RAG (Retrieval-Augmented Generation) bridges that gap without fine-tuning:

1. **Ingestion** — the PDF is parsed into pages, each page is split into overlapping text chunks, and every chunk is embedded into a 384-dimensional vector using a local sentence-transformer model.
2. **Storage** — the vectors and their source text are stored in a ChromaDB collection on disk, keyed to that specific PDF.
3. **Retrieval** — when you ask a question, the question is embedded using the same model and the four most semantically similar chunks are retrieved from ChromaDB.
4. **Generation** — those four chunks are inserted verbatim into a strict prompt that instructs the LLM to answer *only* from the provided context. The LLM never sees the rest of the PDF, only the most relevant excerpts.
5. **Citation** — the page numbers from the retrieved chunks are extracted from metadata and displayed alongside the answer.

The key constraint: if the retrieved chunks do not contain the answer, the LLM is instructed to respond with exactly `"Not found in the document."` — and the application displays that fallback verbatim with no citations.

---

## Project Structure

```
pdf_qa_assistant/
├── app.py                  # Streamlit entry point — UI and session state
├── config.py               # Environment variable loading (GROQ_API_KEY)
├── pdf_loader.py           # PDF validation and page extraction
├── text_splitter.py        # Overlapping chunk generation with metadata propagation
├── embeddings.py           # HuggingFace embedder factory
├── vector_store.py         # ChromaDB collection lifecycle and retriever factory
├── qa_chain.py             # Prompt construction, LLM invocation, fallback logic
├── citations.py            # Page citation formatting
├── requirements.txt        # Pinned dependencies
├── .env.example            # Template for required environment variables
├── .env                    # Your local secrets (not committed)
├── chroma_db/              # ChromaDB persistence directory (auto-created)
└── tests/
    ├── __init__.py
    ├── test_integration_smoke.py    # End-to-end pipeline smoke test
    └── test_vector_store_bugfix.py  # ChromaDB collection lifecycle unit tests
```

---

## Local Setup

### Prerequisites

- Python 3.11 or 3.13
- A [Groq API key](https://console.groq.com) (free tier, no credit card required)

### 1. Clone the repository

```bash
git clone https://github.com/your-username/pdf_qa_assistant.git
cd pdf_qa_assistant
```

### 2. Create and activate a virtual environment

**macOS / Linux**
```bash
python -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

The first `pip install` will also download the `sentence-transformers/all-MiniLM-L6-v2` model weights (~90 MB) from HuggingFace Hub. Subsequent runs use the local cache.

### 4. Configure environment variables

Copy the example file and add your Groq API key:

```bash
cp .env.example .env
```

Edit `.env`:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get your key at [console.groq.com](https://console.groq.com). The free tier provides ample quota for personal use.

### 5. Run the application

```bash
python -m streamlit run app.py
```

Streamlit will open the app in your browser at `http://localhost:8501`.

**First run note:** the embedding model loads on startup (~5–10 seconds). It is cached in session state for the rest of your session.

---

## Running Tests

Run the full test suite from the project root:

```bash
python -m pytest tests/ -v
```

**What the tests cover:**

- `test_integration_smoke.py` — end-to-end pipeline test using a synthetic in-memory PDF. The LLM is mocked (no Groq API call); the HuggingFace embedder runs locally. ChromaDB storage is isolated to a temporary directory and auto-cleaned.
- `test_vector_store_bugfix.py` — unit tests for the ChromaDB collection lifecycle: first-time upload (no existing collection), re-upload (replace existing), genuine error propagation, retriever creation, and collection naming.

The integration smoke test takes ~30 seconds on first run (embedding model load). Subsequent runs within the same process are faster.

---

## Hallucination Fallback

The prompt sent to the LLM includes an explicit instruction:

> *If the answer is not present in the context, respond with exactly this string and nothing else: `Not found in the document.`*

The application checks the response at display time: if the answer equals this exact string, citations are suppressed. This means:

- The LLM is never allowed to draw on its pre-training knowledge to fill gaps.
- Vague or partially relevant answers are still possible if the LLM misinterprets the context — this is a best-effort constraint, not a hard semantic guarantee.
- If ChromaDB returns zero chunks (empty collection or retrieval failure), the fallback is returned immediately without calling the LLM at all.

---

## Deployment — Streamlit Community Cloud

Streamlit Community Cloud provides free hosting for public GitHub repositories.

1. Push your repository to GitHub (public or private).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app**, select your repository, set the main file to `app.py`.
4. Open **Advanced settings → Secrets** and add:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   ```
5. Click **Deploy**.

Streamlit Cloud injects secrets as environment variables; `config.py` reads `GROQ_API_KEY` from the environment automatically.

**Note on ChromaDB persistence:** Streamlit Community Cloud uses an ephemeral filesystem. The `chroma_db/` directory will be recreated on each restart. Users will need to re-upload their PDF after each cold start.

---

## Security

> **Never commit `.env` to version control.**

`.env` contains your `GROQ_API_KEY`. Exposing this key in a public repository could result in unauthorized API usage billed to your account.

Ensure `.env` is listed in `.gitignore`:

```
.env
```

When deploying, always use the host platform's secrets manager (Streamlit Cloud Secrets, environment variables, etc.) — never hardcode keys in source files.

---

## Future Improvements

- **Multi-PDF session** — allow querying across multiple uploaded documents simultaneously
- **Streaming responses** — stream the LLM output token-by-token using Streamlit's `write_stream`
- **Chunk size controls** — expose chunk size and overlap as sidebar sliders for power users
- **Persistent sessions** — save conversation history across browser sessions using a lightweight database
- **Alternative LLMs** — add model selector (e.g. `llama-3.3-70b-versatile`, `mixtral-8x7b-32768`) via Groq's model list
- **PDF text highlighting** — highlight the exact retrieved passages in a PDF viewer
- **Metadata filters** — let users restrict retrieval to specific page ranges
- **Docker packaging** — add a `Dockerfile` for containerised local deployment
