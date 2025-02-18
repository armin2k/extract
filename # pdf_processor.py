# pdf_processor.py

import logging
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
import os
from ocr_utils import preprocess_image, post_process_text

# Configure Poppler path (adjust if needed)
POPPLER_PATH = os.getenv("POPPLER_PATH", "/opt/homebrew/bin")

def extract_text(pdf_path: str) -> str:
    """
    Extract text from a PDF page by page using PyPDF2.
    If extraction is insufficient, fall back to OCR.
    """
    text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
            if len(text.strip()) > 100:
                return text
    except Exception as e:
        logging.error(f"Standard extraction failed: {e}")

    # Fall back to OCR
    try:
        # Increase DPI to improve OCR resolution
        images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH, dpi=300)
        ocr_text = []
        custom_config = "--psm 6 --oem 3"
        for img in images:
            processed_img = preprocess_image(img)
            text_img = pytesseract.image_to_string(processed_img, lang='por', config=custom_config)
            text_img = post_process_text(text_img)
            ocr_text.append(text_img)
        return "\n".join(ocr_text)
    except Exception as e:
        logging.error(f"OCR failed: {e}")
        return ""