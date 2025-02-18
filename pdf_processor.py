# pdf_processor.py
import logging
import os
from typing import List
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
from ocr_utils import preprocess_image, post_process_text

logger = logging.getLogger(__name__)

def extract_text(pdf_path: str) -> List[str]:
    """
    Extract text from a PDF file, returning a list of page texts.
    First attempts extraction with PyPDF2; if not sufficient, falls back to OCR.
    """
    pages_text: List[str] = []
    try:
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            for page in reader.pages:
                text = page.extract_text() or ""
                pages_text.append(text)
            if any(len(text.strip()) > 100 for text in pages_text):
                return pages_text
    except Exception as e:
        logger.exception("Error using PyPDF2 extraction: %s", e)
    
    # Fall back to OCR if PyPDF2 extraction is insufficient.
    try:
        dpi = 300
        poppler_path = os.getenv("POPPLER_PATH", "/opt/homebrew/bin")
        images = convert_from_path(pdf_path, poppler_path=poppler_path, dpi=dpi)
        pages_text = []
        custom_config = "--psm 6 --oem 3"

        # Use a ProcessPoolExecutor for parallel OCR processing (CPU-bound task)
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        def process_image(img):
            try:
                processed = preprocess_image(img)
                text_img = pytesseract.image_to_string(processed, lang='por', config=custom_config)
                return post_process_text(text_img)
            except Exception as e:
                logger.exception("Error processing image: %s", e)
                return ""
        
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_image, img) for img in images]
            for future in as_completed(futures):
                pages_text.append(future.result())
        return pages_text
    except Exception as e:
        logger.exception("Error using OCR extraction: %s", e)
        return []