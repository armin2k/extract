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
        gray = enhancer.enhance(2)  # Adjust factor as needed
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
    If a line contains one of the expected category keywords, remove it and then prepend it to the line.
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
            lines = [line.strip() for line in page_text.splitlines() if line.strip()]
            doc["document"]["pages"].append({"page_number": idx, "lines": lines})
        return json.dumps(doc, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.exception("Error in wrap_pages_in_json: %s", e)
        raise

def extract_company_info_from_text(text: str) -> Dict[str, str]:
    """
    Extract the company name and CNPJ from OCR text.
    
    This function first splits the text into lines and then:
      - If it finds a line with "CNPJ:", it attempts to use the previous line as the company name,
        but only if that line ends with a common corporate suffix (LTDA, SA, or S/A).
      - If that doesn't work, it scans all lines for one that ends with one of these suffixes.
      - The CNPJ is extracted using a regex that captures a common formatted CNPJ.
    
    Returns:
        A dictionary with keys 'company_name' and 'cnpj'.
    """
    try:
        company_name = ""
        cnpj = ""
        
        # Split text into non-empty lines.
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        
        # Define a pattern that matches common company suffixes.
        suffix_pattern = re.compile(r".*(LTDA|S/?A)$", re.IGNORECASE)
        
        # First, if there's a line containing "CNPJ:", try to use the previous line if it ends with a common suffix.
        for i, line in enumerate(lines):
            if "CNPJ:" in line.upper():
                # Attempt to extract CNPJ from this line.
                cnpj_match = re.search(r"CNPJ:\s*([\d]{2}[.\-\/\s]*[\d]{3}[.\-\/\s]*[\d]{3}[.\-\/\s]*[\d]{4}[-\s]*[\d]{2})", line, re.IGNORECASE)
                if cnpj_match:
                    raw_cnpj = cnpj_match.group(1)
                    cnpj_digits = re.sub(r"[^\d]", "", raw_cnpj)
                    if len(cnpj_digits) == 14:
                        cnpj = f"{cnpj_digits[:2]}.{cnpj_digits[2:5]}.{cnpj_digits[5:8]}/{cnpj_digits[8:12]}-{cnpj_digits[12:]}"
                    else:
                        cnpj = cnpj_digits
                # Use the previous line as candidate, if it exists and ends with a common suffix.
                if i > 0:
                    candidate = lines[i - 1]
                    candidate_clean = re.sub(r"^\d+\s*", "", candidate)  # Remove leading digits
                    if suffix_pattern.search(candidate_clean):
                        company_name = candidate_clean.strip()
                        break
        
        # If we haven't found a company name via the above method, scan all lines for a candidate ending with a common suffix.
        if not company_name:
            for line in lines:
                candidate = re.sub(r"^\d+\s*", "", line)
                if suffix_pattern.search(candidate):
                    company_name = candidate.strip()
                    break
        
        return {"company_name": company_name, "cnpj": cnpj}
    except Exception as e:
        logger.exception("Error in extract_company_info_from_text: %s", e)
        return {"company_name": "", "cnpj": ""}

def extract_json_from_text(text: str) -> str:
    """
    Attempt to extract a JSON block from the text.
    This function looks for content enclosed in triple backticks (optionally with 'json')
    and, if not found, tries to extract from the first "{" to the last "}".
    Returns the extracted JSON string if successful, or None if no valid JSON is found.
    """
    import re
    import json

    # Look for triple backticks encapsulating JSON
    candidates = re.findall(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    for candidate in candidates:
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            continue

    # Fallback: Extract from first '{' to last '}'
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        candidate = text[start:end+1]
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            return None
    return None