# backend/utils.py
"""
Utility helpers: extract text (incl. OCR) from PDFs, split text into chunks,
persist embeddings to Parquet, and provide a local Sentence-Transformers
wrapper that matches LangChain’s Embeddings interface.

Works on Windows; runs unchanged on Linux / macOS.
"""
from __future__ import annotations      # PEP 563/649

import io
import os
import sys
import types
from pathlib import Path
from typing import List, Sequence

import fitz                            # PyMuPDF
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytesseract
from PIL import Image
from langchain.embeddings.base import Embeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# ────────────────────────────────────────────────────────────
# Windows-only shim for libs that try to `import pwd`
# ────────────────────────────────────────────────────────────
if sys.platform.startswith("win") and "pwd" not in sys.modules:
    sys.modules["pwd"] = types.ModuleType("pwd")

# ────────────────────────────────────────────────────────────
# Configure Tesseract location (Windows)
# ────────────────────────────────────────────────────────────
_TESS_EXE = (
    os.getenv("TESSERACT_PATH")
    or r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)
if Path(_TESS_EXE).is_file():
    pytesseract.pytesseract.tesseract_cmd = _TESS_EXE

# ────────────────────────────────────────────────────────────
# Optional import path (old vs. new LangChain loaders)
# ────────────────────────────────────────────────────────────
try:
    from langchain_community.document_loaders import PyPDFLoader
except ImportError:
    from langchain.document_loaders import PyPDFLoader  # type: ignore


# =================================================================
# Public helpers
# =================================================================
def extract_text_and_images_from_pdf(
    pdf_path: str | os.PathLike[str],
    ocr_langs: str = "eng",
) -> str:
    """
    Return combined text from a PDF:

    • Embedded PDF text (fast)  
    • Tesseract OCR on every embedded raster image (slow)

    Raises
    ------
    pytesseract.TesseractNotFoundError
        If Tesseract cannot be located.
    """
    pdf_path = str(pdf_path)

    # 1) Native selectable text
    pages = PyPDFLoader(pdf_path).load()
    full_text = "\n".join(p.page_content for p in pages)

    # 2) OCR for embedded images
    doc = fitz.open(pdf_path)
    for page in doc:
        for info in page.get_images(full=True):
            xref = info[0]
            img_bytes = doc.extract_image(xref)["image"]
            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            ocr_text = pytesseract.image_to_string(img, lang=ocr_langs)
            if ocr_text.strip():
                full_text += "\n" + ocr_text

    return full_text


def simple_chunk_text(
    text: str,
    *,
    chunk_size: int = 1_000,
    chunk_overlap: int = 200,
) -> List[str]:
    """
    Split *text* into recursively sized chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_text(text)


def save_embeddings_parquet(
    vectors: Sequence[Sequence[float]],
    chunks: Sequence[str],
    path: str | Path = "embeddings.parquet",
) -> None:
    """
    Persist ``vectors`` and their source ``chunks`` to Parquet for inspection.
    """
    if len(vectors) != len(chunks):
        raise ValueError("vectors and chunks must have identical length")

    df = pd.DataFrame({"chunk": chunks, "vector": vectors})
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, Path(path))


# =================================================================
# Lightweight local embedding model
# =================================================================
class LocalTransformerEmbeddings(Embeddings):
    """
    Sentence-Transformers wrapper implementing LangChain’s Embeddings API.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)

    # Embeddings interface  -------------------------------------------------
    def embed_documents(self, texts: List[str]) -> List[List[float]]:  # type: ignore[override]
        return self.model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> List[float]:  # type: ignore[override]
        return self.model.encode([text])[0].tolist()
