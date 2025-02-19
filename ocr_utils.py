# ocr_utils.py
import re
import json
import logging
from typing import List, Dict
from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger(__name__)

def preprocess_image(img: Image.Image) -> Image.Image:
    """
    Convert the image to grayscale, boost contrast, and apply a median filter.
    This improves OCR accuracy.
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
    For example, replace an uppercase 'O' between digits with '0'.
    """
    try:
        processed_text = re.sub(r'(?<=\d)O(?=\d)', '0', text)
        return processed_text
    except Exception as e:
        logger.exception("Error in post_process_text: %s", e)
        raise

def reorder_line(line: str, categories: List[str]) -> str:
    """
    If a line contains one of the expected category keywords, remove it and
    then prepend it to the line.
    """
    try:
        found = None
        for cat in categories:
            if re.search(re.escape(cat), line, re.IGNORECASE):
                found = cat
                line = re.sub(re.escape(cat), '', line, flags=re.IGNORECASE, count=1)
                break
        if found:
            line = f"{found} {line}"
        return re.sub(r'\s+', ' ', line).strip()
    except Exception as e:
        logger.exception("Error in reorder_line: %s", e)
        raise

def clean_ocr_text(raw_text: str, categories: List[str]) -> str:
    """
    Clean the OCR text by removing URLs and keeping only lines longer than 10 characters
    that contain digits or one of the specified category keywords.
    It also standardizes lines by reordering with the category if found.
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
    Wrap a list of page texts into a JSON structure with page numbers.
    """
    try:
        doc = {"document": {"pages": []}}
        for idx, page_text in enumerate(pages, start=1):
            # Split the page text into lines and remove empty lines.
            lines = [line.strip() for line in page_text.splitlines() if line.strip()]
            doc["document"]["pages"].append({"page_number": idx, "lines": lines})
        return json.dumps(doc, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.exception("Error in wrap_pages_in_json: %s", e)
        raise

def extract_company_info_from_text(text: str) -> Dict[str, str]:
    """
    Improved extraction of company name and CNPJ from OCR text.
    
    For the company name, this function takes the first non-empty line,
    removes any leading numbers and spaces, and treats the remainder as the company name.
    
    For the CNPJ, it uses a regex to capture a standard formatted CNPJ.
    
    Returns:
        A dictionary with keys 'company_name' and 'cnpj'.
    """
    try:
        company_name = ""
        cnpj = ""
        
        # Split the text into lines and take the first non-empty line as a candidate for the company name.
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            # Assume the first line is something like "3013 ALNITAK COMERCIO DE ARTIGOS PARA CASA LTDA"
            first_line = lines[0]
            # Remove any leading digits and spaces (e.g., "3013 ")
            company_name = re.sub(r"^\d+\s*", "", first_line).strip()
        
        # Use regex to extract the CNPJ. This pattern expects the label "CNPJ:" followed by a CNPJ.
        cnpj_pattern = r"CNPJ:\s*([\d]{2}\.[\d]{3}\.[\d]{3}/[\d]{4}-[\d]{2})"
        cnpj_match = re.search(cnpj_pattern, text, re.IGNORECASE)
        if cnpj_match:
            cnpj = cnpj_match.group(1).strip()
        
        return {"company_name": company_name, "cnpj": cnpj}
    except Exception as e:
        logger.exception("Error in extract_company_info_from_text: %s", e)
        return {"company_name": "", "cnpj": ""}