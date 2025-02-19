import re
import json
import logging
from typing import List, Dict
from PIL import Image, ImageEnhance, ImageFilter

logger = logging.getLogger(__name__)

def preprocess_image(img: Image.Image) -> Image.Image:
    try:
        gray = img.convert('L')
        enhancer = ImageEnhance.Contrast(gray)
        gray = enhancer.enhance(2)
        gray = gray.filter(ImageFilter.MedianFilter())
        return gray
    except Exception as e:
        logger.exception("Error in preprocess_image: %s", e)
        raise

def post_process_text(text: str) -> str:
    try:
        processed_text = re.sub(r'(?<=\d)O(?=\d)', '0', text)
        return processed_text
    except Exception as e:
        logger.exception("Error in post_process_text: %s", e)
        raise

def reorder_line(line: str, categories: List[str]) -> str:
    found = None
    for cat in categories:
        if re.search(re.escape(cat), line, re.IGNORECASE):
            found = cat
            line = re.sub(re.escape(cat), '', line, flags=re.IGNORECASE, count=1)
            break
    if found:
        line = f"{found} {line}"
    return re.sub(r'\s+', ' ', line).strip()

def clean_ocr_text(raw_text: str, categories: List[str]) -> str:
    cleaned_lines = []
    for line in raw_text.splitlines():
        line = re.sub(r"http\S+", "", line)
        if len(line) > 10 and (re.search(r'\d', line.lstrip()) or any(cat.lower() in line.lower() for cat in categories)):
            cleaned_lines.append(reorder_line(line, categories))
    return "\n".join(cleaned_lines)

def wrap_pages_in_json(text: str) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    wrapped = {"document": {"lines": lines}}
    return json.dumps(wrapped, ensure_ascii=False, indent=2)

def extract_company_info_from_text(text: str) -> Dict[str, str]:
    try:
        company_name = ""
        cnpj = ""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        suffix_pattern = re.compile(r".*(LTDA|S/?A)$", re.IGNORECASE)
        for i, line in enumerate(lines):
            if "CNPJ:" in line.upper():
                cnpj_match = re.search(r"CNPJ:\s*([\d]{2}[.\-\/\s]*[\d]{3}[.\-\/\s]*[\d]{3}[.\-\/\s]*[\d]{4}[-\s]*[\d]{2})", line, re.IGNORECASE)
                if cnpj_match:
                    raw_cnpj = cnpj_match.group(1)
                    cnpj_digits = re.sub(r"[^\d]", "", raw_cnpj)
                    if len(cnpj_digits) == 14:
                        cnpj = f"{cnpj_digits[:2]}.{cnpj_digits[2:5]}.{cnpj_digits[5:8]}/{cnpj_digits[8:12]}-{cnpj_digits[12:]}"
                    else:
                        cnpj = cnpj_digits
                if i > 0:
                    candidate = lines[i - 1]
                    candidate_clean = re.sub(r"^\d+\s*", "", candidate)
                    if suffix_pattern.search(candidate_clean):
                        company_name = candidate_clean.strip()
                        break
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
    candidates = re.findall(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    for candidate in candidates:
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            continue
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1:
        candidate = text[start:end+1]
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            logger.error("Failed to decode JSON from extracted candidate.")
            return None
    return None