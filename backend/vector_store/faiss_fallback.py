# backend/vector_store/faiss_fallback.py

"""
FAISS-based local vector store — a safety net when Qdrant isn’t reachable.

Core capabilities
-----------------
1. Build / load a flat L2 index (`faiss.IndexFlatL2`).
2. Persist vectors + their raw text to disk.
3. Add new embeddings in batches.
4. k-NN search that returns the best-matching chunks.

The class mirrors the public API of `QdrantStore` so the rest of the
codebase can switch between online and offline stores with minimal fuss.
"""

from __future__ import annotations

import os
import pickle
from typing import List

import numpy as np

try:
    import faiss                   # pip install faiss-cpu
except ModuleNotFoundError as exc:
    raise ImportError(
        "faiss-cpu is not installed. "
        "Run `pip install faiss-cpu` to enable the FAISS fallback store."
    ) from exc


class FaissFallbackStore:
    """
    Lightweight FAISS wrapper.

    Parameters
    ----------
    dim : int
        Embedding dimensionality.
    index_path : str, default "faiss.index"
        Where to save / load the FAISS index file.
    store_path : str, default "chunks.pkl"
        Where to save / load the raw text chunks.
    """

    def __init__(
        self,
        dim: int,
        index_path: str = "faiss.index",
        store_path: str = "chunks.pkl",
    ):
        self.dim = dim
        self.index_path = index_path
        self.store_path = store_path

        # Initialise index and chunk store
        if os.path.exists(index_path) and os.path.exists(store_path):
            self.index = faiss.read_index(index_path)
            with open(store_path, "rb") as f:
                self.chunks: List[str] = pickle.load(f)
        else:
            self.index = faiss.IndexFlatL2(dim)
            self.chunks: List[str] = []

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def add_embeddings(self, embeddings: List[List[float]], chunks: List[str]) -> None:
        """
        Add *embeddings* and their corresponding *chunks* to the store.
        """
        if len(embeddings) != len(chunks):
            raise ValueError("embeddings and chunks must have the same length")

        vectors = np.array(embeddings, dtype="float32")
        self.index.add(vectors)
        self.chunks.extend(chunks)
        self._save()

    def search(self, query_embedding: List[float], top_k: int = 5) -> List[str]:
        """
        Return the *top_k* most similar chunks for *query_embedding*.
        """
        if len(self.chunks) == 0:
            return []

        query = np.asarray([query_embedding], dtype="float32")
        distances, indices = self.index.search(query, top_k)

        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.chunks):
                results.append(self.chunks[idx])
        return results

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _save(self) -> None:
        """
        Persist index + chunks to disk.
        """
        faiss.write_index(self.index, self.index_path)
        with open(self.store_path, "wb") as f:
            pickle.dump(self.chunks, f)
