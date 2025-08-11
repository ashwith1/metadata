# backend/ingestion/extractor.py

import os
import io
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from langchain.document_loaders import PyPDFLoader
from typing import List

def extract_text_and_images_from_pdf(pdf_path: str, ocr_langs: str = "eng+deu") -> str:
    """
    Extracts text (including OCR on images) from a PDF.
    - Uses PyPDFLoader for native text.
    - Runs pytesseract OCR on each embedded image.
    """
    # Load native text
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    full_text = "\n".join([d.page_content for d in docs])

    # Open with PyMuPDF for images
    doc = fitz.open(pdf_path)
    for page in doc:
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            ocr_text = pytesseract.image_to_string(image, lang=ocr_langs)
            full_text += "\n" + ocr_text
    return full_text
