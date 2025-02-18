# ocr_utils.py
import re
import json
import logging
from typing import List
from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger(__name__)

def preprocess_image(img: Image.Image) -> Image.Image:
    """
    Convert image to grayscale, boost contrast, and apply a median filter.
    """
    try:
        gray = img.convert('L')
        enhancer = ImageEnhance.Contrast(gray)
        gray = enhancer.enhance(2)  # Adjust the factor as needed
        gray = gray.filter(ImageFilter.MedianFilter())
        return gray
    except Exception as e:
        logger.exception("Error in preprocess_image: %s", e)
        raise

def post_process_text(text: str) -> str:
    """
    Apply post-processing corrections to OCR text.
    E.g., replace an uppercase 'O' between digits with zero.
    """
    try:
        return re.sub(r'(?<=\d)O(?=\d)', '0', text)
    except Exception as e:
        logger.exception("Error in post_process_text: %s", e)
        raise

def reorder_line(line: str, categories: List[str]) -> str:
    """
    If a line contains a category keyword, remove it and prepend it.
    """
    try:
        found = None
        for cat in categories:
            if re.search(re.escape(cat), line, re.IGNORECASE):
                found = cat
                line = re.sub(re.escape(cat), '', line, flags=re.IGNORECASE, count=1)
                break
        if found:
            line = found + " " + line
        return re.sub(r'\s+', ' ', line).strip()
    except Exception as e:
        logger.exception("Error in reorder_line: %s", e)
        raise

def clean_ocr_text(raw_text: str, categories: List[str]) -> str:
    """
    Clean the OCR text:
      - Remove URLs.
      - Keep only lines longer than 10 characters that include a digit or a category keyword.
      - Reorder lines so that the keyword appears at the beginning.
    """
    try:
        cleaned_lines = []
        for line in raw_text.splitlines():
            line = re.sub(r"http\S+", "", line)  # Remove URLs
            if len(line.strip()) > 10 and (re.search(r'\d', line.strip()) or any(cat.lower() in line.lower() for cat in categories)):
                cleaned_lines.append(reorder_line(line, categories))
        return "\n".join(cleaned_lines)
    except Exception as e:
        logger.exception("Error in clean_ocr_text: %s", e)
        raise

def wrap_pages_in_json(pages: List[str]) -> str:
    """
    Wrap a list of page texts into a structured JSON format with page numbers.
    """
    try:
        doc = {"document": {"pages": []}}
        for idx, page_text in enumerate(pages, start=1):
            lines = [line.strip() for line in page_text.splitlines() if line.strip()]
            doc["document"]["pages"].append({"page_number": idx, "lines": lines})
        return json.dumps(doc, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.exception("Error in wrap_pages_in_json: %s", e)
        raise