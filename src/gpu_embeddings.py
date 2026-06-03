"""GPU-accelerated embedding module using sentence-transformers.

Provides a singleton embedding model that runs on CUDA when available,
replacing TF-IDF for semantic similarity in tool retrieval and memory search.
Falls back to TF-IDF if sentence-transformers is not installed.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL: Any = None
_EMBEDDING_DEVICE: str = "cpu"


def _get_embedding_model(model_name: str = "all-MiniLM-L6-v2", device: str = "cuda"):
    """Lazy-load a singleton SentenceTransformer on the specified device."""
    global _EMBEDDING_MODEL, _EMBEDDING_DEVICE
    if _EMBEDDING_MODEL is not None and _EMBEDDING_DEVICE == device:
        return _EMBEDDING_MODEL
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model %s on %s...", model_name, device)
        _EMBEDDING_MODEL = SentenceTransformer(model_name, device=device)
        _EMBEDDING_DEVICE = device
        dim_fn = getattr(_EMBEDDING_MODEL, "get_embedding_dimension", None) or _EMBEDDING_MODEL.get_sentence_embedding_dimension
        logger.info("Embedding model loaded (dim=%d)", dim_fn())
        return _EMBEDDING_MODEL
    except Exception as e:
        logger.warning("Could not load sentence-transformers model: %s. Falling back to TF-IDF.", e)
        return None


def gpu_cosine_similarity(query: str, corpus: list[str], model_name: str, device: str) -> np.ndarray:
    """Compute cosine similarities between query and corpus using GPU embeddings.

    Returns an array of similarity scores, one per corpus document.
    """
    model = _get_embedding_model(model_name, device)
    if model is None:
        raise RuntimeError("Embedding model not available")

    all_texts = corpus + [query]
    embeddings = model.encode(all_texts, convert_to_numpy=True, show_progress_bar=False)
    doc_embeddings = embeddings[:-1]
    query_embedding = embeddings[-1:]

    norms_docs = np.linalg.norm(doc_embeddings, axis=1, keepdims=True).clip(min=1e-10)
    norm_query = np.linalg.norm(query_embedding, axis=1, keepdims=True).clip(min=1e-10)
    doc_embeddings = doc_embeddings / norms_docs
    query_embedding = query_embedding / norm_query

    sims = (query_embedding @ doc_embeddings.T).flatten()
    return sims


def is_gpu_embeddings_available() -> bool:
    """Check if sentence-transformers is importable."""
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False
