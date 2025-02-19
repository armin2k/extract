# pdf_processor.py
import logging
import os
from typing import List
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
from ocr_utils import preprocess_image, post_process_text

logger = logging.getLogger(__name__)

def process_page(img) -> str:
    """
    Process a single image page:
      - Preprocess the image for better OCR accuracy.
      - Extract text using Tesseract.
      - Post-process the text.
    
    Returns:
      The extracted text as a string.
    """
    try:
        processed_img = preprocess_image(img)
        # Use Tesseract with a custom configuration for layout analysis.
        custom_config = "--psm 6 --oem 3"
        text = pytesseract.image_to_string(processed_img, lang='por', config=custom_config)
        return post_process_text(text)
    except Exception as e:
        logger.exception("Error processing page: %s", e)
        return ""

def extract_text(pdf_path: str) -> List[str]:
    """
    Extract text from a PDF file.

    1. First, attempt extraction using PyPDF2.
    2. If the extracted text is insufficient (e.g., most pages are empty),
       fall back to OCR processing:
         - Convert PDF pages to images using pdf2image.
         - Process the images concurrently using a ProcessPoolExecutor.

    Returns:
      A list of text strings (one per page).
    """
    pages_text = []
    try:
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            for page in reader.pages:
                text = page.extract_text() or ""
                pages_text.append(text)
            # If at least one page contains a significant amount of text, return it.
            if any(len(t.strip()) > 100 for t in pages_text):
                logger.info("Text extracted using PyPDF2.")
                return pages_text
    except Exception as e:
        logger.exception("Error using PyPDF2 extraction: %s", e)
    
    # Fall back to OCR extraction if PyPDF2 didn't yield sufficient text.
    try:
        poppler_path = os.getenv("POPPLER_PATH", "/opt/homebrew/bin")
        # Increase dpi for better OCR quality if needed.
        images = convert_from_path(pdf_path, poppler_path=poppler_path, dpi=300)
        pages_text = []
        from concurrent.futures import ProcessPoolExecutor, as_completed
        max_workers = os.cpu_count() or 1
        logger.info("Processing OCR on %d pages using %d workers.", len(images), max_workers)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_page, img) for img in images]
            for future in as_completed(futures):
                pages_text.append(future.result())
        logger.info("Text extracted using OCR.")
        return pages_text
    except Exception as e:
        logger.exception("Error using OCR extraction: %s", e)
        return []