# backend/graphs/rag_graph.py
from typing import Optional, Dict

from backend.vector_store.qdrant_store import QdrantStore
from backend.llm.rag_chain import build_rag_chain
from backend.graphs.monitor import traced_span

def run_rag_graph(
    *,
    question: str,
    collection_name: str = "pdf_docs",
    llm_model: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
    top_k: int = 5,
    span_attrs: Optional[Dict[str, str]] = None,
) -> str:
    attrs = {"llm_model": llm_model, "collection": collection_name, "action": "rag"}
    if span_attrs:
        attrs.update(span_attrs)

    with traced_span("rag_pipeline", **attrs):
        with traced_span("retrieve_and_answer", top_k=top_k, question=question, **attrs):
            store = QdrantStore(collection_name=collection_name)
            retriever = store.as_retriever(search_kwargs={"k": top_k})
            rag_chain = build_rag_chain(llm_model=llm_model, retriever=retriever)
            result = rag_chain.run({"query": question})
            return result
