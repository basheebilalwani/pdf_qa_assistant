"""
embeddings.py — HuggingFace embedder factory.

Provides one public function:
  - get_embedder: returns a HuggingFaceEmbeddings instance configured to use
    the sentence-transformers/all-MiniLM-L6-v2 model, running entirely locally.

The model is downloaded from HuggingFace Hub on first use and cached locally
by the sentence-transformers library for subsequent runs. No external embedding
API is called at any point.

Callers should cache the returned instance (e.g. in Streamlit session_state)
to avoid reloading the model on every call.
"""

from langchain_huggingface import HuggingFaceEmbeddings

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedder() -> HuggingFaceEmbeddings:
    """
    Return a HuggingFaceEmbeddings instance using
    model_name="sentence-transformers/all-MiniLM-L6-v2".

    The model runs entirely on the local machine — no external embedding API
    is called. On first use the model files are downloaded from HuggingFace
    Hub and cached locally by the sentence-transformers library.

    This function may be called multiple times; callers should cache the
    result to avoid repeated model loads.
    """
    return HuggingFaceEmbeddings(model_name=_MODEL_NAME)
