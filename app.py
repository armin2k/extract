import os
import re
import json
import math
import time
import datetime
import requests
import pandas as pd
import logging
import concurrent.futures

from flask import Flask, request, render_template_string, send_from_directory, url_for, redirect
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from PIL import Image, ImageEnhance, ImageFilter
from pdf2image import convert_from_path
from PyPDF2 import PdfReader
import pytesseract
from tenacity import retry, wait_exponential, stop_after_attempt

# For image pre-processing enhancements
import cv2
import numpy as np

# SQLAlchemy imports for PostgreSQL integration
from sqlalchemy import create_engine, Column, Integer, String, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# -------------------------------
# Load configuration and set up logging
# -------------------------------
load_dotenv()

APP_VERSION = "v1.0.1"  # Displayed in the browser footer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Environment variables for external tools and database
POPPLER_PATH = os.getenv("POPPLER_PATH", "/opt/homebrew/bin")
SCALE_FACTOR = int(os.getenv("SCALE_FACTOR", 1))
CATEGORIES_FILE = "categories.json"
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://armin@localhost/balancesheetdb")

# -------------------------------
# SQLAlchemy setup
# -------------------------------
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class BalanceSheetAnalysis(Base):
    __tablename__ = "balance_sheet_analysis"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)  # Original filename
    company_name = Column(String, index=True)
    cnpj = Column(String, index=True)
    analysis_data = Column(JSON)
    ocr_data = Column(JSON)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

# Create the table if it doesn't exist
Base.metadata.create_all(bind=engine)

# -------------------------------
# Load Categories and precompile regex patterns
# -------------------------------
def load_categories() -> list:
    try:
        with open(CATEGORIES_FILE, encoding="utf-8") as f:
            data = json.load(f)
            return data["categories"]
    except Exception as e:
        logging.error(f"Failed to load categories: {e}")
        return []

CATEGORIES = load_categories()
CATEGORY_PATTERNS = [(cat, re.compile(re.escape(cat), re.IGNORECASE)) for cat in CATEGORIES]

# -------------------------------
# Image Pre-processing Functions
# -------------------------------
def deskew_image(img: Image.Image) -> Image.Image:
    try:
        img_np = np.array(img.convert('L'))
        coords = np.column_stack(np.where(img_np > 0))
        if coords.size == 0:
            return img
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        (h, w) = img_np.shape
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(img_np, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return Image.fromarray(rotated)
    except Exception as e:
        logging.error(f"Deskewing failed: {e}")
        return img

def preprocess_image(img: Image.Image) -> Image.Image:
    try:
        img = deskew_image(img)
    except Exception as e:
        logging.error(f"Error during deskewing: {e}")
    gray = img.convert('L')
    enhancer = ImageEnhance.Contrast(gray)
    gray = enhancer.enhance(2)
    gray = gray.filter(ImageFilter.MedianFilter())
    img_np = np.array(gray)
    bin_img = cv2.adaptiveThreshold(
        img_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    return Image.fromarray(bin_img)

def post_process_text(text: str) -> str:
    text = re.sub(r'(?<=\d)O(?=\d)', '0', text)
    return text

# -------------------------------
# PDF and OCR Processing Functions
# -------------------------------
def extract_text(pdf_path: str) -> str:
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
    try:
        images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH, dpi=300)
        ocr_text = []
        custom_config = "--psm 6 --oem 3"
        for idx, img in enumerate(images):
            processed_img = preprocess_image(img)
            text_img = pytesseract.image_to_string(processed_img, lang='por', config=custom_config)
            text_img = post_process_text(text_img)
            ocr_text.append(text_img)
            logging.info(f"OCR completed for page {idx+1}.")
        return "\n".join(ocr_text)
    except Exception as e:
        logging.error(f"OCR failed: {e}")
        return ""

def reorder_line(line: str, categories: list) -> str:
    found = None
    for cat, pattern in CATEGORY_PATTERNS:
        if pattern.search(line):
            found = cat
            line = pattern.sub('', line, count=1)
            break
    if found:
        line = found + " " + line
    return re.sub(r'\s+', ' ', line).strip()

def clean_ocr_text(raw_text: str, categories: list) -> str:
    cleaned_lines = []
    for line in raw_text.splitlines():
        line = re.sub(r"http\S+", "", line)
        if len(line) > 10 and (re.search(r'\d', line.lstrip()) or any(cat.lower() in line.lower() for cat in categories)):
            cleaned_lines.append(reorder_line(line, categories))
    return "\n".join(cleaned_lines)

def wrap_text_in_json(text: str) -> str:
    lines = []
    for idx, line in enumerate(text.splitlines()):
        if line.strip():
            lines.append({"line_number": idx+1, "text": line.strip()})
    wrapped = {"document": {"lines": lines}}
    return json.dumps(wrapped, ensure_ascii=False, indent=2)

# -------------------------------
# Data Parsing Functions
# -------------------------------
def parse_value(value) -> float:
    if value in [None, "NaN", ""]:
        return math.nan
    if isinstance(value, str):
        value = value.replace('R$', '').strip()
        value = value.replace('–', '-').replace('—', '-')
        negative = False
        if value.startswith('-'):
            negative = True
            value = value[1:].strip()
        if value.startswith('(') and value.endswith(')'):
            negative = True
            value = value[1:-1].strip()
        value = value.replace(' ', '')
        value = value.replace('.', '').replace(',', '.')
        try:
            num = float(value)
        except Exception:
            return math.nan
        if negative:
            num = -num
        return num * SCALE_FACTOR
    try:
        return float(value) * SCALE_FACTOR
    except Exception:
        return math.nan

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
            logging.error("Failed to decode JSON from extracted candidate.")
            return None
    return None

def format_financial_data(response_json: dict, categories: list) -> dict:
    try:
        content = response_json.get('choices', [{}])[0].get('message', {}).get('content', '')
        if not content.strip():
            logging.error("Formatting error: API returned empty content.")
            return None
        json_text = extract_json_from_text(content)
        if not json_text:
            logging.error("Formatting error: Could not extract JSON content from the response.")
            logging.error("Raw response snippet: %s", content[:500])
            return None
        try:
            raw_data = json.loads(json_text)
        except json.JSONDecodeError as jde:
            logging.error("Formatting error: Failed to decode JSON. Error: %s", jde)
            return None
        if any(key in categories for key in raw_data.keys()):
            raw_data = {"Ano Desconhecido": raw_data}
        formatted = {}
        for year, data in raw_data.items():
            formatted[year] = {}
            for category in categories:
                raw_value = data.get(category, math.nan)
                formatted[year][category] = parse_value(raw_value)
        return formatted
    except Exception as e:
        logging.error(f"Formatting error: {e}")
        return None

# -------------------------------
# Checksum Validation Function
# -------------------------------
def validate_checksums(data: dict) -> dict:
    checksum_report = {}
    tolerance = 0.01
    for year, values in data.items():
        report = {}
        ativo_total = values.get("Ativo Total")
        if ativo_total is None or math.isnan(ativo_total):
            report["ativo"] = "Ativo Total is missing or NaN"
        else:
            sum_ativo = 0.0
            ativo_found = False
            for key, value in values.items():
                if key.startswith("Ativo") and key != "Ativo Total":
                    if not math.isnan(value):
                        sum_ativo += value
                        ativo_found = True
            if ativo_found:
                if abs(sum_ativo - ativo_total) > tolerance:
                    report["ativo"] = f"Checksum error: sum of Ativo items is {sum_ativo}, expected {ativo_total}."
                else:
                    report["ativo"] = "OK"
            else:
                report["ativo"] = "No Ativo line items found for checksum validation"
        passivo_total = values.get("Passivo Total")
        if passivo_total is None or math.isnan(passivo_total):
            report["passivo"] = "Passivo Total is missing or NaN"
        else:
            sum_passivo = 0.0
            passivo_found = False
            for key, value in values.items():
                if key.startswith("Passivo") and key != "Passivo Total":
                    if not math.isnan(value):
                        sum_passivo += value
                        passivo_found = True
            if passivo_found:
                if abs(sum_passivo - passivo_total) > tolerance:
                    report["passivo"] = f"Checksum error: sum of Passivo items is {sum_passivo}, expected {passivo_total}."
                else:
                    report["passivo"] = "OK"
            else:
                report["passivo"] = "No Passivo line items found for checksum validation"
        checksum_report[year] = report
    return checksum_report

# -------------------------------
# API Integration Functions with Retry
# -------------------------------
def get_api_parameters(provider: str) -> tuple:
    if provider == "deepseek":
        url = "http://localhost:11434/v1/chat/completions"
        headers = {}
        model = "deepseek-r1:14b"
        timeout_value = 240
    else:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            logging.error("OPENAI_API_KEY is not set in environment variables.")
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}"}
        model = "gpt-4o-mini"
        timeout_value = 180
    return url, headers, model, timeout_value

@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
def make_api_call(url: str, headers: dict, payload: dict, timeout_value: int) -> dict:
    response = requests.post(url, headers=headers, json=payload, timeout=timeout_value)
    if response.status_code != 200:
        raise Exception(f"API call failed with status code {response.status_code}: {response.text}")
    return response.json()

def analyze_with_api(text_json: str, provider: str, categories: list) -> dict:
    prompt = f"""
Você é um especialista em contabilidade brasileira. A seguir, é fornecido um balanço patrimonial extraído por OCR, com a formatação preservada num objeto JSON.
Extraia somente os dados financeiros relevantes, convertendo "1.234,56" para 1234.56 e "(1.234,56)" para -1234.56; use NaN para valores ausentes.
Se houver dados de múltiplos anos, utilize os anos (4 dígitos) como chaves; caso contrário, utilize "Ano Desconhecido".
Retorne APENAS um objeto JSON com os valores extraídos.

Categorias:
{json.dumps(CATEGORIES, indent=4, ensure_ascii=False)}

Documento (em JSON):
{text_json}
    """
    url, headers, model, timeout_value = get_api_parameters(provider)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1
    }
    try:
        response_json = make_api_call(url, headers, payload, timeout_value)
        return format_financial_data(response_json, CATEGORIES)
    except Exception as e:
        logging.error(f"API Error: {e}")
        return None

def merge_analysis_results(results: list, categories: list) -> dict:
    merged = {}
    for result in results:
        if not result:
            continue
        for year, data in result.items():
            if year not in merged:
                merged[year] = data
            else:
                for category in categories:
                    current_val = merged[year].get(category, math.nan)
                    new_val = data.get(category, math.nan)
                    if math.isnan(current_val) and not math.isnan(new_val):
                        merged[year][category] = new_val
    return merged

# -------------------------------
# Batch Processing Functions
# -------------------------------
def create_batches(doc: str, batch_size: int, overlap_lines: int) -> list:
    lines = doc.splitlines()
    batches = []
    start = 0
    while start < len(lines):
        char_count = 0
        end = start
        while end < len(lines) and (char_count + len(lines[end]) <= batch_size):
            char_count += len(lines[end])
            end += 1
        batch = "\n".join(lines[start:end])
        batches.append(batch)
        start = max(0, end - overlap_lines)
    return batches

def analyze_document_in_batches(text_json: str, provider: str, categories: list, batch_size: int = 10000, overlap_lines: int = 3) -> tuple:
    logs = []
    try:
        doc_data = json.loads(text_json)
        if isinstance(doc_data.get("document"), dict) and "lines" in doc_data["document"]:
            doc = "\n".join(line["text"] for line in doc_data["document"]["lines"])
        else:
            doc = doc_data.get("document", "")
    except Exception as e:
        logs.append(f"Error loading wrapped JSON: {e}")
        return None, "\n".join(logs)
    batches = create_batches(doc, batch_size, overlap_lines)
    logs.append(f"Created {len(batches)} batches (batch_size={batch_size}, overlap_lines={overlap_lines}).")
    results = []
    total_batches = len(batches)
    max_workers = min(32, (os.cpu_count() or 1) * 2)
    logs.append(f"Processing {total_batches} batches concurrently (max_workers={max_workers}).")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(analyze_with_api, wrap_text_in_json(batch), provider, CATEGORIES): i + 1
            for i, batch in enumerate(batches)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            i = future_to_index[future]
            try:
                result = future.result()
                if result:
                    results.append(result)
                    logs.append(f"Batch {i} processed successfully.")
                else:
                    logs.append(f"Warning: No result from batch {i}.")
            except Exception as exc:
                logs.append(f"Batch {i} generated an exception: {exc}")
    if results:
        merged_result = merge_analysis_results(results, CATEGORIES)
        logs.append("Merged results from batches successfully.")
        return merged_result, "\n".join(logs)
    else:
        logs.append("No results were obtained from any batches.")
        return None, "\n".join(logs)

# -------------------------------
# Company Info Extraction
# -------------------------------
def extract_company_info(ocr_text: str, filename: str) -> dict:
    """
    Attempt to extract the company name and CNPJ from the OCR text.
    
    - For CNPJ: Looks for a pattern like 'XX.XXX.XXX/0001-XX'.
    - For company name: Searches for keywords ("Razão Social", "Empresa", "Entidade")
      followed by a separator and captures text ending with common suffixes (LTDA, S/A, or SA).
    - If not found, falls back to using the filename:
      It extracts the substring from the start of the filename until the first digit.
    """
    # Pattern to extract CNPJ
    cnpj_pattern = r'\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}'
    cnpj_match = re.search(cnpj_pattern, ocr_text)
    cnpj = cnpj_match.group() if cnpj_match else None

    # Pattern to extract company name using keywords and legal suffixes
    company_pattern = r'(?:Razão Social|Empresa|Entidade)[:\-]\s*([^\n]+?(?:LTDA|Ltda|ltda|S/A|s/a|SA|sa))'
    company_match = re.search(company_pattern, ocr_text, re.IGNORECASE)
    if company_match:
        company_name = company_match.group(1).strip()
    else:
        # Fallback: use the filename
        base = os.path.splitext(filename)[0]  # remove .pdf extension
        # Extract from the start of the filename until the first digit
        fallback_match = re.search(r'^([^\d]+)', base)
        if fallback_match:
            company_name = fallback_match.group(1).strip()
        else:
            company_name = base.strip()

    return {"company_name": company_name, "cnpj": cnpj}

# -------------------------------
# Flask Web Application and New Routes
# -------------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs("output", exist_ok=True)

# Landing page with navigation
UPLOAD_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Balance Sheet Analyzer</title>
  <style>
    body { font-family: Arial, sans-serif; background-color: #f2f2f2; margin: 0; padding: 0; }
    .navbar { background-color: #333; overflow: hidden; }
    .navbar a { float: left; display: block; color: #f2f2f2; text-align: center; padding: 14px 16px; text-decoration: none; }
    .navbar a:hover { background-color: #ddd; color: black; }
    .container { width: 80%; margin: 50px auto; background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
    footer { text-align: center; margin-top: 20px; font-size: 14px; color: #777; }
  </style>
  <script>
    function showProgress() {
      document.getElementById("progressDiv").style.display = "block";
    }
  </script>
</head>
<body>
<div class="navbar">
  <a href="/">Upload</a>
  <a href="/search_page">Search</a>
</div>
<div class="container">
  <h2>Balance Sheet Analyzer - Upload a PDF</h2>
  <form method="post" enctype="multipart/form-data" action="/upload" onsubmit="showProgress()">
    <label for="file">Select a Balance Sheet PDF:</label>
    <input type="file" name="file" id="file">
    <label for="provider">Select API Provider:</label>
    <select name="provider" id="provider">
      <option value="chatgpt">ChatGPT API (gpt-4o-mini)</option>
      <option value="deepseek">DeepSeek API (via Ollama offline)</option>
    </select>
    <input type="submit" value="Upload">
  </form>
  <div class="progress" id="progressDiv" style="display:none;">
    <p>Processing... please wait.</p>
  </div>
</div>
<footer>Version: {{ version }}</footer>
</body>
</html>
"""

# Search form page
SEARCH_PAGE = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Search Balance Sheets</title>
  <style>
    body { font-family: Arial, sans-serif; background-color: #f2f2f2; margin: 0; padding: 0; }
    .navbar { background-color: #333; overflow: hidden; }
    .navbar a { float: left; display: block; color: #f2f2f2; text-align: center; padding: 14px 16px; text-decoration: none; }
    .navbar a:hover { background-color: #ddd; color: black; }
    .container { width: 80%; margin: 50px auto; background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
    footer { text-align: center; margin-top: 20px; font-size: 14px; color: #777; }
  </style>
</head>
<body>
<div class="navbar">
  <a href="/">Upload</a>
  <a href="/search_page">Search</a>
</div>
<div class="container">
  <h2>Search Balance Sheets</h2>
  <form method="get" action="/search_results">
    <label for="company">Company Name:</label>
    <input type="text" id="company" name="company" placeholder="Enter company name">
    <br><br>
    <label for="cnpj">CNPJ:</label>
    <input type="text" id="cnpj" name="cnpj" placeholder="Enter CNPJ">
    <br><br>
    <input type="submit" value="Search">
  </form>
</div>
<footer>Version: {{ version }}</footer>
</body>
</html>
"""

RESULT_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Upload Result</title>
  <style>
    body { font-family: Arial, sans-serif; background-color: #f2f2f2; margin: 0; padding: 0; }
    .navbar { background-color: #333; overflow: hidden; }
    .navbar a { float: left; display: block; color: #f2f2f2; text-align: center; padding: 14px 16px; text-decoration: none; }
    .navbar a:hover { background-color: #ddd; color: black; }
    .container { width: 80%; margin: 50px auto; background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); text-align: center; }
    footer { text-align: center; margin-top: 20px; font-size: 14px; color: #777; }
  </style>
</head>
<body>
<div class="navbar">
  <a href="/">Upload</a>
  <a href="/search_page">Search</a>
</div>
<div class="container">
  <h2>Upload Result for {{ filename }}</h2>
  {% if download_analysis_link %}
    <p><a href="{{ download_analysis_link }}">Download Analysis JSON</a></p>
  {% else %}
    <p>No analysis result available.</p>
  {% endif %}
  <p><a href="{{ download_ocr_link }}">Download OCR JSON</a></p>
  {% if download_xls_link %}
    <p><a href="{{ download_xls_link }}">Download Analysis XLS</a></p>
  {% endif %}
  <h3>Batch Processing Logs</h3>
  <pre>{{ batch_logs }}</pre>
  <br>
  <a href="/">Upload another file</a>
</div>
<footer>Version: {{ version }}</footer>
</body>
</html>
"""

# Search results page
SEARCH_RESULTS = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Search Results</title>
  <style>
    body { font-family: Arial, sans-serif; background-color: #f2f2f2; margin: 0; padding: 0; }
    .navbar { background-color: #333; overflow: hidden; }
    .navbar a { float: left; display: block; color: #f2f2f2; text-align: center; padding: 14px 16px; text-decoration: none; }
    .navbar a:hover { background-color: #ddd; color: black; }
    .container { width: 90%; margin: 50px auto; background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
    table { width: 100%; border-collapse: collapse; }
    th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
    th { background-color: #4CAF50; color: white; }
    footer { text-align: center; margin-top: 20px; font-size: 14px; color: #777; }
  </style>
</head>
<body>
<div class="navbar">
  <a href="/">Upload</a>
  <a href="/search_page">Search</a>
</div>
<div class="container">
  <h2>Search Results</h2>
  {% if results %}
  <table>
    <tr>
      <th>ID</th>
      <th>Company Name</th>
      <th>CNPJ</th>
      <th>Last Updated</th>
      <th>Details</th>
    </tr>
    {% for r in results %}
    <tr>
      <td>{{ r.id }}</td>
      <td>{{ r.company_name }}</td>
      <td>{{ r.cnpj }}</td>
      <td>{{ r.created_at }}</td>
      <td><a href="/record/{{ r.id }}">View Details</a></td>
    </tr>
    {% endfor %}
  </table>
  {% else %}
    <p>No records found.</p>
  {% endif %}
  <br>
  <a href="/search_page">Back to Search</a>
</div>
<footer>Version: {{ version }}</footer>
</body>
</html>
"""

# Record detail page
RECORD_DETAIL = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Record Detail</title>
  <style>
    body { font-family: Arial, sans-serif; background-color: #f2f2f2; margin: 0; padding: 0; }
    .navbar { background-color: #333; overflow: hidden; }
    .navbar a { float: left; display: block; color: #f2f2f2; text-align: center; padding: 14px 16px; text-decoration: none; }
    .navbar a:hover { background-color: #ddd; color: black; }
    .container { width: 90%; margin: 50px auto; background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
    /* Tab styling */
    .tab { overflow: hidden; border-bottom: 1px solid #ccc; }
    .tab button { background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; transition: 0.3s; font-size: 17px; }
    .tab button:hover { background-color: #ddd; }
    .tab button.active { background-color: #ccc; }
    .tabcontent { display: none; padding: 20px 0; }
    table { width: 100%; border-collapse: collapse; margin-top: 20px; }
    th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }
    th { background-color: #4CAF50; color: white; }
    footer { text-align: center; margin-top: 20px; font-size: 14px; color: #777; }
  </style>
  <script>
    function openTab(evt, tabName) {
      var i, tabcontent, tablinks;
      tabcontent = document.getElementsByClassName("tabcontent");
      for (i = 0; i < tabcontent.length; i++) {
        tabcontent[i].style.display = "none";
      }
      tablinks = document.getElementsByClassName("tablinks");
      for (i = 0; i < tablinks.length; i++) {
        tablinks[i].className = tablinks[i].className.replace(" active", "");
      }
      document.getElementById(tabName).style.display = "block";
      evt.currentTarget.className += " active";
    }
    window.onload = function() {
      document.getElementsByClassName("tablinks")[0].click();
    }
  </script>
</head>
<body>
<div class="navbar">
  <a href="/">Upload</a>
  <a href="/search_page">Search</a>
</div>
<div class="container">
  <h2>Record Detail (ID: {{ record.id }})</h2>
  <p><strong>Company Name:</strong> {{ record.company_name }}</p>
  <p><strong>CNPJ:</strong> {{ record.cnpj }}</p>
  <p><strong>Last Updated:</strong> {{ record.created_at }}</p>
  
  <!-- Tab navigation -->
  <div class="tab">
    <button class="tablinks" onclick="openTab(event, 'balance')">Balance Sheet</button>
    <button class="tablinks" onclick="openTab(event, 'checksum')">Checksum</button>
  </div>
  
  <!-- Balance Sheet Tab -->
  <div id="balance" class="tabcontent">
    <h3>Extracted Balance Sheet Information</h3>
    {% if analysis and sorted_years %}
    <table>
      <tr>
        <th>Category</th>
        {% for year in sorted_years %}
        <th>{{ year }}</th>
        {% endfor %}
      </tr>
      {% for category in categories %}
      <tr>
        <td>{{ category }}</td>
        {% for year in sorted_years %}
        <td>{{ analysis[year].get(category, "NaN") }}</td>
        {% endfor %}
      </tr>
      {% endfor %}
    </table>
    {% else %}
      <p>No analysis data available.</p>
    {% endif %}
  </div>
  
  <!-- Checksum Tab -->
  <div id="checksum" class="tabcontent">
    <h3>Checksum Information</h3>
    {% if checksum and sorted_years %}
    <table>
      <tr>
        <th>Type</th>
        {% for year in sorted_years %}
        <th>{{ year }}</th>
        {% endfor %}
      </tr>
      <tr>
        <td>Ativo</td>
        {% for year in sorted_years %}
        <td>{{ checksum[year].get("ativo", "N/A") }}</td>
        {% endfor %}
      </tr>
      <tr>
        <td>Passivo</td>
        {% for year in sorted_years %}
        <td>{{ checksum[year].get("passivo", "N/A") }}</td>
        {% endfor %}
      </tr>
    </table>
    {% else %}
      <p>No checksum data available.</p>
    {% endif %}
  </div>
  
  <h3>Downloads</h3>
  <ul>
    <li><a href="{{ download_analysis_link }}">Download Analysis JSON</a></li>
    <li><a href="{{ download_xls_link }}">Download Analysis XLS</a></li>
    <li><a href="{{ download_ocr_link }}">Download OCR JSON</a></li>
  </ul>
  <br>
  <a href="/search_page">Back to Search</a>
</div>
<footer>Version: {{ version }}</footer>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(UPLOAD_HTML, version=APP_VERSION)

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return "No file part", 400
    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400
    if not file.filename.lower().endswith(".pdf"):
        return "Only PDF files are allowed.", 400

    provider = request.form.get("provider", "chatgpt")
    filename = secure_filename(file.filename)
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(upload_path)
    
    logging.info(f"File {filename} saved to {upload_path}.")

    raw_text = extract_text(upload_path)
    cleaned_text = clean_ocr_text(raw_text, CATEGORIES)
    wrapped_json = wrap_text_in_json(cleaned_text)
    
    json_text_path = os.path.join("output", f"{filename}_ocr.json")
    try:
        with open(json_text_path, "w", encoding="utf-8") as f:
            f.write(wrapped_json)
    except Exception as e:
        logging.error(f"Error writing OCR JSON file: {e}")
    
    if len(cleaned_text) > 10000:
        result, batch_logs = analyze_document_in_batches(wrapped_json, provider, CATEGORIES, batch_size=10000, overlap_lines=3)
    else:
        result = analyze_with_api(wrapped_json, provider, CATEGORIES)
        batch_logs = "Single API call used (no batch processing)."
    
    download_analysis_link = None
    download_xls_link = None
    if result:
        checksum_report = validate_checksums(result)
        output_data = {
            "financial_data": result,
            "checksum_report": checksum_report
        }
        analysis_json_path = os.path.join("output", f"{filename}_analysis.json")
        try:
            with open(analysis_json_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False, default=lambda x: "NaN" if math.isnan(x) else x)
        except Exception as e:
            logging.error(f"Error writing analysis JSON file: {e}")
        
        try:
            with pd.ExcelWriter(os.path.join("output", f"{filename}_analysis.xlsx")) as writer:
                df_financial = pd.DataFrame.from_dict(result, orient="index")
                df_financial.index.name = "Year"
                df_financial = df_financial.transpose()
                df_financial.to_excel(writer, sheet_name="Financial Data")
                
                df_checksum = pd.DataFrame.from_dict(checksum_report, orient="index")
                df_checksum.index.name = "Year"
                df_checksum = df_checksum.transpose()
                df_checksum.to_excel(writer, sheet_name="Checksum Report")
        except Exception as e:
            logging.error(f"Error writing analysis XLS file: {e}")
        
        download_analysis_link = url_for("download_file", filename=f"{filename}_analysis.json")
        download_ocr_link = url_for("download_file", filename=f"{filename}_ocr.json")
        download_xls_link = url_for("download_file", filename=f"{filename}_analysis.xlsx")
    else:
        download_ocr_link = url_for("download_file", filename=f"{filename}_ocr.json")
        batch_logs += "\nNo analysis result obtained."

    company_info = extract_company_info(raw_text, filename)

    session = SessionLocal()
    try:
        record = BalanceSheetAnalysis(
            filename=filename,
            company_name=company_info.get("company_name"),
            cnpj=company_info.get("cnpj"),
            analysis_data=result if result else {},
            ocr_data=json.loads(wrapped_json)
        )
        session.add(record)
        session.commit()
        logging.info(f"Record saved in DB for company: {company_info.get('company_name')}, CNPJ: {company_info.get('cnpj')}")
    except Exception as e:
        session.rollback()
        logging.error(f"Error saving record to DB: {e}")
    finally:
        session.close()
    
    return render_template_string(RESULT_HTML,
                                  filename=filename,
                                  download_analysis_link=download_analysis_link,
                                  download_ocr_link=download_ocr_link,
                                  download_xls_link=download_xls_link,
                                  batch_logs=batch_logs,
                                  version=APP_VERSION)

@app.route("/download/<path:filename>")
def download_file(filename: str):
    return send_from_directory("output", filename, as_attachment=True)

# New route: Search form page
@app.route("/search_page", methods=["GET"])
def search_page():
    return render_template_string(SEARCH_PAGE, version=APP_VERSION)

# New route: Display search results in HTML
@app.route("/search_results", methods=["GET"])
def search_results():
    company_query = request.args.get("company")
    cnpj_query = request.args.get("cnpj")
    session = SessionLocal()
    try:
        query = session.query(BalanceSheetAnalysis)
        if company_query:
            query = query.filter(BalanceSheetAnalysis.company_name.ilike(f"%{company_query}%"))
        if cnpj_query:
            query = query.filter(BalanceSheetAnalysis.cnpj.ilike(f"%{cnpj_query}%"))
        results = query.all()
        result_data = []
        for res in results:
            result_data.append({
                "id": res.id,
                "company_name": res.company_name,
                "cnpj": res.cnpj,
                "created_at": res.created_at.strftime("%Y-%m-%d %H:%M:%S")
            })
        return render_template_string(SEARCH_RESULTS, results=result_data, version=APP_VERSION)
    except Exception as e:
        logging.error(f"Error during search: {e}")
        return f"Error during search: {e}", 500
    finally:
        session.close()

# New route: Record detail page
@app.route("/record/<int:record_id>", methods=["GET"])
def record_detail(record_id):
    session = SessionLocal()
    try:
        record = session.query(BalanceSheetAnalysis).filter(BalanceSheetAnalysis.id == record_id).first()
        if not record:
            return "Record not found", 404
        analysis = record.analysis_data if record.analysis_data else {}
        sorted_years = sorted(analysis.keys()) if analysis else []
        checksum = validate_checksums(analysis) if analysis else {}
        return render_template_string(RECORD_DETAIL,
                                      record=record,
                                      analysis=analysis,
                                      sorted_years=sorted_years,
                                      checksum=checksum,
                                      categories=CATEGORIES,
                                      download_analysis_link=url_for("download_file", filename=f"{record.filename}_analysis.json"),
                                      download_xls_link=url_for("download_file", filename=f"{record.filename}_analysis.xlsx"),
                                      download_ocr_link=url_for("download_file", filename=f"{record.filename}_ocr.json"),
                                      version=APP_VERSION)
    except Exception as e:
        logging.error(f"Error fetching record detail: {e}")
        return f"Error fetching record detail: {e}", 500
    finally:
        session.close()

if __name__ == "__main__":
    app.run(debug=True)