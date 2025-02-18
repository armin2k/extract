# pdf_processor.py

import logging
import os
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
from ocr_utils import preprocess_image, post_process_text

# Ensure POPPLER_PATH is set (adjust if needed)
POPPLER_PATH = os.getenv("POPPLER_PATH", "/opt/homebrew/bin")

def extract_text(pdf_path: str) -> list:
    """
    Extract text from a PDF page by page using PyPDF2.
    If extraction is insufficient, fall back to OCR.
    Returns a list of page texts.
    """
    pages_text = []
    try:
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                pages_text.append(page_text)
            # If any page has sufficient text, assume extraction succeeded.
            if any(len(text.strip()) > 100 for text in pages_text):
                return pages_text
    except Exception as e:
        logging.error(f"Standard extraction failed: {e}")

    # Fall back to OCR if PyPDF2 extraction is not sufficient.
    try:
        images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH, dpi=300)
        pages_text = []
        custom_config = "--psm 6 --oem 3"
        for img in images:
            processed_img = preprocess_image(img)
            text_img = pytesseract.image_to_string(processed_img, lang='por', config=custom_config)
            text_img = post_process_text(text_img)
            pages_text.append(text_img)
        return pages_text
    except Exception as e:
        logging.error(f"OCR failed: {e}")
        return []