# ocr_utils.py

import re
import json
from PIL import Image, ImageEnhance, ImageFilter

def preprocess_image(img: Image.Image) -> Image.Image:
    """
    Convert image to grayscale, boost contrast, and apply a median filter.
    This helps improve OCR accuracy.
    """
    gray = img.convert('L')
    enhancer = ImageEnhance.Contrast(gray)
    gray = enhancer.enhance(2)  # Adjust as needed
    gray = gray.filter(ImageFilter.MedianFilter())
    return gray

def post_process_text(text: str) -> str:
    """
    Apply post-processing corrections to the OCR text.
    For example, replace common misinterpretations (like "O" in numeric contexts).
    """
    text = re.sub(r'(?<=\d)O(?=\d)', '0', text)
    return text

def reorder_line(line: str, categories: list) -> str:
    """
    If a line contains one of the expected category keywords,
    remove it from its current position and prepend it.
    """
    found = None
    for cat in categories:
        if re.search(re.escape(cat), line, re.IGNORECASE):
            found = cat
            line = re.sub(re.escape(cat), '', line, flags=re.IGNORECASE, count=1)
            break
    if found:
        line = found + " " + line
    return re.sub(r'\s+', ' ', line).strip()

def clean_ocr_text(raw_text: str, categories: list) -> str:
    """
    Clean the OCR text:
      - Remove URLs.
      - Keep only lines longer than 10 characters that contain at least one digit 
        or one of the expected category keywords.
      - Reorder lines so the category keyword appears at the beginning.
    """
    cleaned_lines = []
    for line in raw_text.splitlines():
        line = re.sub(r"http\S+", "", line)  # Remove URLs
        if len(line) > 10 and (re.search(r'\d', line.lstrip()) or any(cat.lower() in line.lower() for cat in categories)):
            cleaned_lines.append(reorder_line(line, categories))
    return "\n".join(cleaned_lines)

def wrap_pages_in_json(pages: list) -> str:
    """
    Wrap a list of page texts into a JSON structure.
    Each page becomes an object with its page number and its list of nonempty lines.
    """
    doc = {"document": {"pages": []}}
    for idx, page_text in enumerate(pages, start=1):
        # Split page text into lines and remove any empty lines.
        lines = [line.strip() for line in page_text.splitlines() if line.strip()]
        doc["document"]["pages"].append({"page_number": idx, "lines": lines})
    return json.dumps(doc, ensure_ascii=False, indent=2)