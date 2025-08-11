# app.py

import os
import streamlit as st

# CRITICAL: PyTorch-Streamlit compatibility fix
def fix_torch_streamlit_compatibility():
    try:
        import torch
        torch.classes.__path__ = []
        print("✅ PyTorch-Streamlit compatibility fix applied")
    except Exception as e:
        print(f"⚠️ PyTorch fix warning: {e}")

fix_torch_streamlit_compatibility()

# Configuration loading from Streamlit secrets
def load_config():
    """Load configuration from Streamlit secrets and environment variables"""
    
    # Helper function to get values from secrets or environment
    def get_secret(key, default=None):
        # Check environment variables first
        env_value = os.getenv(key)
        if env_value:
            return env_value
        
        # Then check Streamlit secrets
        try:
            if hasattr(st, 'secrets') and key in st.secrets:
                return st.secrets[key]
        except Exception:
            pass
        
        return default
    
    # Set required environment variables from secrets
    secrets_to_env = {
        "TOGETHER_API_KEY": get_secret("TOGETHER_API_KEY"),
        "LANGFUSE_PUBLIC_KEY": get_secret("LANGFUSE_PUBLIC_KEY"),
        "LANGFUSE_SECRET_KEY": get_secret("LANGFUSE_SECRET_KEY"),
        "LANGFUSE_HOST": get_secret("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        "QDRANT_API_KEY": get_secret("QDRANT_API_KEY"),
        "QDRANT_URL": get_secret("QDRANT_URL", "http://localhost:6333")
    }
    
    # Set environment variables
    for key, value in secrets_to_env.items():
        if value:
            os.environ[key] = str(value)

# Load configuration before other imports
load_config()

# NOW import the rest (remove the old dotenv loading)
import json
import uuid
import time
import hashlib
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, List
import atexit

# Continue with your existing imports...
from backend.utils import extract_text_and_images_from_pdf, simple_chunk_text
from backend.embeddings.embedder import LocalTransformerEmbeddings
from backend.vector_store.qdrant_store import QdrantStore
from backend.graphs.metadata_graph import run_metadata_graph
from backend.graphs.rag_graph import run_rag_graph
from backend.graphs.monitor import get_callback, traced_span, begin_trace

# ───────────────── configuration ─────────────────
load_dotenv(os.path.join("config", ".env"))
st.set_page_config(
    page_title="PDF Metadata Extractor & Chatbot",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📘 PDF Metadata Extractor & Chatbot")
st.caption("Extract module metadata, index your document, and chat with it — with Langfuse tracing.")

# Ensure Langfuse exists
cb = get_callback()
if cb is None:
    st.error("Langfuse keys missing. Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in config/.env.")
    st.stop()

# ───────────────── session ─────────────────
os.makedirs("tmp", exist_ok=True)

def _init_session():
    ss = st.session_state
    ss.setdefault("conversation_id", str(uuid.uuid4()))
    ss.setdefault("pdf_tmp_path", os.path.join("tmp", f"{ss['conversation_id']}.pdf"))
    ss.setdefault("history", [])
    ss.setdefault("metadata", None)
    ss.setdefault("uploaded_pdf_bytes", None)
    ss.setdefault("llm_sel", "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free")
    ss.setdefault("pdf_text_cache", "")
    ss.setdefault("has_pdf", False)
    ss.setdefault("has_index", False)
    ss.setdefault("last_index_time", None)
    ss.setdefault("chunk_count", 0)
    ss.setdefault("current_pdf_hash", None)  # Track if PDF changed

_init_session()

# Single Langfuse trace per session
begin_trace(
    trace_name=st.session_state["conversation_id"],
    app="pdf-metadata-rag",
    started_at=str(datetime.utcnow()),
)

# Consistent collection name for both indexing and QA
COLLECTION_NAME = "pdf_docs"

# NEW ─────────────────────────────────────────────────────
def _drop_collection():
    """Guaranteed clean-up when Streamlit run terminates (stop/refresh)."""
    try:
        QdrantStore(collection_name=COLLECTION_NAME).delete_collection()
    except Exception:
        pass

atexit.register(_drop_collection)
# The handler fires when …
# 1️⃣ Streamlit server stops (Ctrl-C / container exit)
# 2️⃣ The script is re-executed (browser refresh / widget triggers)

# ───────────────── helpers ─────────────────
def get_pdf_hash() -> str:
    """Generate hash of current PDF to detect when it changes"""
    if st.session_state.get("uploaded_pdf_bytes"):
        return hashlib.md5(st.session_state["uploaded_pdf_bytes"]).hexdigest()
    return ""

def check_existing_index() -> bool:
    """Check if there's already an index for the current PDF in Qdrant/FAISS"""
    try:
        store = QdrantStore(collection_name=COLLECTION_NAME)
        if store._use_faiss:
            return hasattr(store, "_faiss") and len(store._faiss.chunks) > 0
        else:
            # Check if collection exists and has vectors
            collections = store._client.get_collections()
            collection_names = {c.name for c in collections.collections}
            if COLLECTION_NAME in collection_names:
                collection_info = store._client.get_collection(COLLECTION_NAME)
                return getattr(collection_info, "vectors_count", 0) > 0
        return False
    except Exception:
        return False

def add_message(role: str, content: str, meta: Optional[Dict[str, Any]] = None):
    st.session_state["history"].append(
        {"role": role, "content": content, "ts": time.time(), "meta": meta or {}}
    )

def run_metadata():
    if not st.session_state["pdf_text_cache"]:
        st.warning("Load or index the PDF first.")
        return
    try:
        with traced_span(
            "metadata_request",
            conversation_id=st.session_state["conversation_id"],
            action="metadata",
        ):
            md = run_metadata_graph(
                text=st.session_state["pdf_text_cache"],
                llm_model=st.session_state["llm_sel"],
                span_attrs={"conversation_id": st.session_state["conversation_id"]},
            )
        st.session_state["metadata"] = md
        add_message("assistant", "Metadata extracted. Toggle 'Table view' to inspect or download below.", {"type": "metadata"})
    except Exception as e:
        st.error(f"Metadata extraction failed: {e}")
        add_message("assistant", f"Metadata extraction failed: {e}")

def ensure_indexing() -> str:
    if not st.session_state["has_pdf"]:
        return "Upload a PDF first."
    
    try:
        with traced_span("indexing", conversation_id=st.session_state["conversation_id"]):
            # Extract text if not cached
            if not st.session_state["pdf_text_cache"]:
                st.session_state["pdf_text_cache"] = extract_text_and_images_from_pdf(
                    st.session_state["pdf_tmp_path"]
                )
            
            text = st.session_state["pdf_text_cache"]
            if not text.strip():
                st.session_state["has_index"] = False
                return "No selectable or OCR text found in PDF."

            chunks = simple_chunk_text(text, chunk_size=1000, chunk_overlap=200)
            if not chunks:
                st.session_state["has_index"] = False
                return "Failed to create text chunks from PDF."

            store = QdrantStore(collection_name=COLLECTION_NAME)
            embedder = LocalTransformerEmbeddings()
            store.upsert_documents(chunks, embedder)

            # Verification with actual retrieval test
            try:
                test_docs = store.similarity_search("test", embedder, k=1)
                if len(test_docs) == 0:
                    st.session_state["has_index"] = False
                    return "Indexing verification failed: No documents retrievable."
            except Exception as e:
                st.session_state["has_index"] = False
                return f"Indexing verification failed: {e}"

            # SUCCESS: Update all state variables
            st.session_state["has_index"] = True
            st.session_state["chunk_count"] = len(chunks)
            st.session_state["last_index_time"] = datetime.utcnow().isoformat(timespec="seconds")
            st.session_state["current_pdf_hash"] = get_pdf_hash()
            
            return f"✅ Indexed successfully! Created {len(chunks)} chunks."
            
    except Exception as e:
        st.session_state["has_index"] = False
        st.session_state["chunk_count"] = 0
        return f"❌ Indexing failed: {str(e)}"

def answer_question(question: str) -> str:
    if not st.session_state["has_pdf"]:
        return "Please upload a PDF first."
    
    # Check if we have an index OR if there's an existing index in the store
    if not st.session_state["has_index"]:
        # Double-check if there's actually an existing index
        if check_existing_index():
            st.session_state["has_index"] = True
            st.session_state["chunk_count"] = "unknown"
            st.session_state["last_index_time"] = "previous session"
        else:
            return "Please run indexing before asking questions."
    
    try:
        with traced_span(
            "chat_question",
            conversation_id=st.session_state["conversation_id"],
            action="qa",
            question=question,
        ):
            reply = run_rag_graph(
                question=question,
                llm_model=st.session_state["llm_sel"],
                collection_name=COLLECTION_NAME,
                span_attrs={"conversation_id": st.session_state["conversation_id"]},
            )
        return reply
    except Exception as e:
        # Don't reset has_index on Q&A errors - the index is still valid
        return f"Error answering question: {str(e)}"

def render_feedback_controls(idx: int):
    try:
        fb = st.feedback("thumbs", key=f"fb_{idx}")
        if fb is not None:
            st.session_state["history"][idx]["meta"]["feedback"] = fb
            if fb == "👎":
                note = st.text_input("Tell us what to improve (optional)", key=f"fb_note_{idx}")
                if note:
                    st.session_state["history"][idx]["meta"]["feedback_note"] = note
    except Exception:
        c1, c2 = st.columns(2)
        with c1:
            if st.button("👍 Helpful", key=f"up_{idx}"):
                st.session_state["history"][idx]["meta"]["feedback"] = "up"
        with c2:
            if st.button("👎 Not helpful", key=f"down_{idx}"):
                st.session_state["history"][idx]["meta"]["feedback"] = "down"

def render_edit_controls(idx: int, content: str):
    with st.expander("Edit assistant response"):
        edited = st.text_area("Edit text", value=content, height=200, key=f"edit_{idx}")
        if st.button("Validate & save", key=f"save_{idx}"):
            edited = edited.strip()
            if not edited:
                st.error("Response cannot be empty.")
            else:
                st.session_state["history"][idx]["content"] = edited
                st.success("Saved.")

def suggested_questions() -> List[str]:
    base = [
        "Metadata",
        "Give a high-level summary of this PDF.",
        "What are the assessment methods?",
        "List key dates and deadlines.",
        "Who are the instructors and what are their contact details?",
        "What are the learning outcomes?",
        "Are there any prerequisites or required readings?",
    ]
    if st.session_state.get("metadata"):
        base.insert(0, "Show the extracted metadata in a concise list.")
    
    # Only show "Run indexing" if we definitely don't have an index
    if st.session_state["has_pdf"] and not st.session_state["has_index"] and not check_existing_index():
        base.insert(0, "Run indexing")
    
    return base[:6]

def handle_suggestion_click(q: str):
    add_message("user", q)
    if q.lower() == "run indexing":
        msg = ensure_indexing()
        add_message("assistant", msg)
    elif q.lower() in {"metadata", "show metadata"}:
        run_metadata()
    else:
        ans = answer_question(q)
        add_message("assistant", ans)
    st.rerun()

# ───────────────── sidebar ─────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    st.session_state["llm_sel"] = st.selectbox(
        "LLM Model",
        [
            "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
        ],
        index=[
            "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",
        ].index(st.session_state["llm_sel"]),
        help="Together.ai model used for extraction and QA.",
        key="sel_model",
    )

    st.divider()
    st.subheader("Upload")
    up = st.file_uploader("Drag and drop PDF", type=["pdf"], accept_multiple_files=False, key="upload_pdf")
    if up is not None:
        b = up.read()
        new_hash = hashlib.md5(b).hexdigest() if b else ""
        
        # Only reset if it's a different PDF
        if new_hash != st.session_state.get("current_pdf_hash"):
            st.session_state["uploaded_pdf_bytes"] = b
            with open(st.session_state["pdf_tmp_path"], "wb") as f:
                f.write(b)
            st.session_state["has_pdf"] = True
            st.session_state["has_index"] = False  # Only reset on new PDF
            st.session_state["metadata"] = None
            st.session_state["pdf_text_cache"] = ""
            st.session_state["chunk_count"] = 0
            st.session_state["current_pdf_hash"] = new_hash
            st.success(f"Uploaded: {up.name}")

    if st.button("Preview PDF", key="btn_preview_pdf_sidebar"):
        if not st.session_state["has_pdf"]:
            st.warning("Upload a PDF first.")
        else:
            with traced_span("preview_pdf", conversation_id=st.session_state["conversation_id"]):
                if not st.session_state["pdf_text_cache"]:
                    st.session_state["pdf_text_cache"] = extract_text_and_images_from_pdf(
                        st.session_state["pdf_tmp_path"]
                    )
            preview = st.session_state["pdf_text_cache"][:2000]
            st.text_area("Preview (first 2,000 chars)", preview, height=200, key="preview_area_sidebar")

    st.divider()
    st.subheader("Indexing")
    if st.button("Run indexing & extraction", key="btn_run_indexing_sidebar"):
        if not st.session_state["has_pdf"]:
            st.warning("Upload a PDF first.")
        else:
            with st.spinner("Indexing PDF..."):
                msg = ensure_indexing()
            st.write(msg)

    # Enhanced status display with double-check
    if st.session_state.get("has_index") or check_existing_index():
        if not st.session_state.get("has_index"):
            # Update state if we found an existing index
            st.session_state["has_index"] = True
            st.session_state["chunk_count"] = "recovered"
            st.session_state["last_index_time"] = "detected"
        
        chunk_info = f" ({st.session_state.get('chunk_count', 'unknown')} chunks)"
        st.success(f"✅ Index ready • {st.session_state.get('last_index_time', '')}{chunk_info}")
    elif st.session_state.get("has_pdf"):
        st.info("📄 PDF loaded — run indexing to enable chat.")
    else:
        st.info("⏳ Waiting for PDF.")

    st.divider()
    st.subheader("Session tools")
    # IMPORTANT: unique keys and also delete vector collection
    if st.button("Clear chat", key="btn_clear_chat_sidebar"):
        st.session_state["history"] = []
        QdrantStore(COLLECTION_NAME).delete_collection()
        st.session_state["has_index"] = False
        st.session_state["chunk_count"] = 0
        st.info("Chat cleared and collection deleted.")
    if st.button("Reset indexing", key="btn_reset_indexing_sidebar"):
        QdrantStore(COLLECTION_NAME).delete_collection()
        st.session_state["has_index"] = False
        st.session_state["chunk_count"] = 0
        st.info("Index reset and collection deleted. You can re-run indexing.")
    st.caption(f"Session ID: {st.session_state['conversation_id'][:8]}")

# ───────────────── main layout ─────────────────
col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("Chat")

    # Steps panel with enhanced status check
    with st.container(border=True):
        st.markdown("### Steps")
        step1 = "✅" if st.session_state["has_pdf"] else "①"
        
        # Enhanced check for step 2
        has_working_index = st.session_state["has_index"] or check_existing_index()
        step2 = "✅" if has_working_index else "②"
        
        st.write(f"{step1} Upload a PDF")
        st.write(f"{step2} Run indexing & extraction")
        
        if not has_working_index:
            st.caption("Tip: Use the sidebar button or click 'Run indexing' below.")
        else:
            st.caption("Index is ready — ask questions below.")

    # Render chat history
    for i, msg in enumerate(st.session_state["history"]):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                render_feedback_controls(i)
                render_edit_controls(i, msg["content"])

with col_right:
    st.subheader("Metadata")
    md = st.session_state.get("metadata")
    if md:
        table_view = st.toggle("Table view", value=True, key="toggle_table_view")
        if table_view:
            df = pd.DataFrame(
                [{"Field": k, "Value": ("" if v is None else v)} for k, v in md.items()]
            )
            st.dataframe(df, use_container_width=True, hide_index=True, height=280)
        else:
            st.json(md)
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "metadata.json",
                data=json.dumps(md, indent=2),
                file_name="metadata.json",
                mime="application/json",
                key="dl_md_json",
            )
        with c2:
            df = pd.DataFrame([md])
            st.download_button(
                "metadata.csv",
                data=df.to_csv(index=False).encode("utf-8"),
                file_name="metadata.csv",
                mime="text/csv",
                key="dl_md_csv",
            )
    else:
        st.info("No metadata yet. Click 'Metadata' in chat or the sidebar after uploading/indexing.")

    st.divider()
    st.subheader("Status")
    
    # Enhanced status display
    has_working_index = st.session_state["has_index"] or check_existing_index()
    index_status = "ready" if has_working_index else "not ready"
    pdf_status = "loaded" if st.session_state["has_pdf"] else "not loaded"
    
    st.write(f"**Index:** {index_status}")
    st.write(f"**PDF:** {pdf_status}")

# ───────────────── suggested questions ─────────────────
st.markdown("#### Suggested questions")
for i, q in enumerate(suggested_questions()):
    st.button(q, key=f"sugg_{i}", on_click=handle_suggestion_click, args=(q,))

# ───────────────── chat input ─────────────────
user_input = st.chat_input("Ask about the document, or type: 'run indexing', 'metadata', 'preview pdf'", key="chat_input_main")

if user_input:
    add_message("user", user_input)
    cmd = user_input.strip().lower()
    assistant_text: Optional[str] = None

    if cmd in {"preview pdf", "preview", "show pdf"}:
        if not st.session_state["has_pdf"]:
            assistant_text = "Upload a PDF first."
        else:
            with traced_span("preview_pdf_chat", conversation_id=st.session_state["conversation_id"]):
                if not st.session_state["pdf_text_cache"]:
                    st.session_state["pdf_text_cache"] = extract_text_and_images_from_pdf(
                        st.session_state["pdf_tmp_path"]
                    )
            assistant_text = st.session_state["pdf_text_cache"][:2000] or "No text found."
    elif cmd in {"run indexing", "index", "indexing"}:
        assistant_text = ensure_indexing()
    elif cmd in {"metadata", "show metadata"}:
        run_metadata()
        assistant_text = None
    else:
        assistant_text = answer_question(user_input)

    if assistant_text is not None:
        add_message("assistant", assistant_text)
    st.rerun()
