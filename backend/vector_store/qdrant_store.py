# backend/vector_store/qdrant_store.py
from __future__ import annotations
import os
from typing import Iterable, List, Dict, Any, Optional

from qdrant_client import QdrantClient, models as qdrant
from langchain_community.vectorstores import Qdrant as LCQdrant
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
import streamlit as st

from backend.vector_store.faiss_fallback import FaissFallbackStore
from backend.utils import save_embeddings_parquet

_QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
_QDRANT_API = os.getenv("QDRANT_API_KEY")

class QdrantStore:
    def __init__(
        self,
        collection_name: str,
        *,
        distance: str = "Cosine",
        host: str = _QDRANT_URL,
        api_key: Optional[str] = _QDRANT_API,
        db_path: str = "qdrant_data",
        faiss_index_path: str = "faiss.index",
        faiss_store_path: str = "chunks.pkl",
    ) -> None:
        self.collection = collection_name
        self._use_faiss = False

        dist_map = {
            "Cosine": qdrant.Distance.COSINE,
            "Euclidean": qdrant.Distance.EUCLID,
            "Dot": qdrant.Distance.DOT,
        }
        if distance not in dist_map:
            raise ValueError(f"distance must be one of {list(dist_map)}")
        self._dist = dist_map[distance]

        try:
            self._client = QdrantClient(url=host, api_key=api_key, timeout=3)
            self._client.get_collections()
            self._embedded = False
        except Exception:
            try:
                self._client = QdrantClient(path=db_path)
                self._client.get_collections()
                self._embedded = True
            except Exception:
                self._use_faiss = True
                self._faiss = FaissFallbackStore(
                    dim=384,
                    index_path=faiss_index_path,
                    store_path=faiss_store_path,
                )

        self._lc: Optional[LCQdrant] = None
        self._last_embedder: Optional[Embeddings] = None

    def _ensure_lc(self, embedder: Embeddings) -> None:
        if self._use_faiss:
            return
        # If wrapper exists, nothing to do
        if self._lc is not None:
            return
        # Ensure collection exists and bind wrapper
        try:
            collections = self._client.get_collections()
            names = {c.name for c in collections.collections}
            if self.collection not in names:
                self._client.create_collection(
                    collection_name=self.collection,
                    vectors_config=qdrant.VectorParams(size=384, distance=self._dist),
                )
            self._lc = LCQdrant(
                client=self._client,
                collection_name=self.collection,
                embeddings=embedder,
            )
            self._last_embedder = embedder
        except Exception as e:
            raise RuntimeError(f"Failed to init LCQdrant for '{self.collection}': {e}")

    def upsert_documents(
        self,
        chunks: Iterable[str],
        embedder: Embeddings,
        metadata: Optional[Dict[str, Any]] = None,
        batch_size: int = 32,
    ) -> None:
        chunks = list(chunks)
        if not chunks:
            return

        vectors = embedder.embed_documents(chunks)
        save_embeddings_parquet(vectors, chunks)

        if self._use_faiss:
            self._faiss.add_embeddings(vectors, chunks)
            return

        docs = [Document(page_content=c, metadata=metadata or {}) for c in chunks]
        self._ensure_lc(embedder)

        try:
            for i in range(0, len(docs), batch_size):
                self._lc.add_documents(docs[i : i + batch_size])  # type: ignore[union-attr]
        except Exception as e:
            st.error(f"❌ Qdrant upsert failed: {e}")
            raise

    def similarity_search(
        self,
        query: str,
        embedder: Embeddings,
        k: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[Document]:
        if self._use_faiss:
            qvec = embedder.embed_query(query)
            texts = self._faiss.search(qvec, top_k=k)
            return [Document(page_content=t) for t in texts]

        self._ensure_lc(embedder)
        try:
            hits = self._lc.similarity_search_with_score(query, k=k)  # type: ignore[union-attr]
            if score_threshold is not None:
                hits = [h for h in hits if h[1] <= score_threshold]
            return [d for d, _ in hits]
        except Exception as e:
            st.error(f"❌ Search failed: {e}")
            return []

    def as_retriever(self, **kwargs):
        if self._use_faiss:
            # Minimal FAISS retriever
            from backend.embeddings.model_registry import get_embedder
            emb = get_embedder("local-mini")
            class _FaissRetriever:
                def __init__(self, store, embedder, kw): 
                    self.s, self.e, self.kw = store, embedder, kw
                def get_relevant_documents(self, q):
                    return self.s.similarity_search(q, self.e, **self.kw)
            return _FaissRetriever(self, emb, kwargs)

        # For Qdrant, bind LC wrapper lazily with any embedder (won't be used for encoding at query time)
        from backend.embeddings.model_registry import get_embedder
        emb = self._last_embedder or get_embedder("local-mini")
        self._ensure_lc(emb)
        return self._lc.as_retriever(**kwargs)  # type: ignore[union-attr]
    

# ────────────────────────────────────────────────────────────
# NEW: collection maintenance helpers
# ────────────────────────────────────────────────────────────
    def collection_exists(self) -> bool:
        """
        True ↔ the current collection already lives in Qdrant/FAISS and is non-empty
        """
        try:
            if self._use_faiss:
                return hasattr(self, "_faiss") and len(self._faiss.chunks) > 0
            names = {c.name for c in self._client.get_collections().collections}
            if self.collection not in names:
                return False
            return self._client.get_collection(self.collection).vectors_count > 0
        except Exception:
            return False

    def delete_collection(self) -> bool:
        """
        Drop the whole collection (or remove the local-FAISS index files).

        Returns
        -------
        bool
            True if the collection (or files) were actually removed.
        """
        try:
            if self._use_faiss:
                if hasattr(self, "_faiss"):
                    import os
                    for p in (self._faiss.index_path, self._faiss.store_path):
                        if os.path.exists(p):
                            os.remove(p)
                    # start afresh
                    self._faiss = FaissFallbackStore(dim=384,
                                                     index_path=self._faiss.index_path,
                                                     store_path=self._faiss.store_path)
                return True

            names = {c.name for c in self._client.get_collections().collections}
            if self.collection in names:
                self._client.delete_collection(self.collection)
            # housekeeping
            self._lc = None
            self._last_embedder = None
            return True
        except Exception as exc:
            print(f"[QdrantStore] Failed to delete collection: {exc}")
            return False
