# backend/embeddings/embedder.py

"""
LangChain-compatible embedding wrapper.

• Supports a local SentenceTransformer model (default)  
• Can transparently switch to Together AI or OpenAI embeddings via
  backend/embeddings/model_registry.py
"""

from __future__ import annotations

from typing import List, Sequence
from functools import lru_cache
import os

from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer


class LocalTransformerEmbeddings(Embeddings):
    """
    Fast, no-API-key embeddings using SentenceTransformers.

    Example
    -------
    >>> embedder = LocalTransformerEmbeddings("all-MiniLM-L6-v2")
    >>> vectors = embedder.embed_documents(["hello world", "good-bye"])
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = _get_cached_st_model(model_name)

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:  # type: ignore[override]
        """
        Embed a list of texts. Returns a list of vector lists (float32).
        """
        return self._model.encode(
            list(texts),
            show_progress_bar=False,
            normalize_embeddings=True,
        ).tolist()

    def embed_query(self, text: str) -> List[float]:  # type: ignore[override]
        """
        Embed a single query string.
        """
        return self._model.encode(
            [text],
            show_progress_bar=False,
            normalize_embeddings=True,
        )[0].tolist()


@lru_cache(maxsize=4)
def _get_cached_st_model(model_name: str) -> SentenceTransformer:
    """
    Load a SentenceTransformer model once and cache it for reuse.

    The cache is keyed by *model_name*, so multiple different models
    can coexist without repeated downloads.
    """
    # Allow overriding the Hugging Face cache directory via env var
    hf_cache = os.getenv("HF_HOME")
    if hf_cache:
        os.environ["HF_HOME"] = hf_cache
        os.environ["HF_DATASETS_CACHE"] = hf_cache
        os.environ["TRANSFORMERS_CACHE"] = hf_cache

    return SentenceTransformer(model_name)
