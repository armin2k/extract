import os
import json
import math
import time
import requests
import re
import pandas as pd  # For converting JSON to XLS
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
from dotenv import load_dotenv
import concurrent.futures
import logging
from flask import Flask, request, render_template_string, send_from_directory, url_for
from werkzeug.utils import secure_filename  # For secure file names
from PIL import Image, ImageEnhance, ImageFilter  # For image pre-processing
from tenacity import retry, wait_exponential, stop_after_attempt  # For exponential backoff

# -------------------------------
# Configuration and Logging Setup
# -------------------------------
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# External configuration values from .env file
POPPLER_PATH = os.getenv("POPPLER_PATH", "/opt/homebrew/bin")
SCALE_FACTOR = int(os.getenv("SCALE_FACTOR", 1))
CATEGORIES_FILE = "categories.json"

def load_categories() -> list:
    """Load categories from a JSON file."""
    try:
        with open(CATEGORIES_FILE, encoding="utf-8") as f:
            data = json.load(f)
            return data["categories"]
    except Exception as e:
        logging.error(f"Failed to load categories: {e}")
        return []

CATEGORIES = load_categories()

# Precompile regex patterns for categories to speed up matching
CATEGORY_PATTERNS = [(cat, re.compile(re.escape(cat), re.IGNORECASE)) for cat in CATEGORIES]

# -------------------------------
# Image Pre-processing Functions
# -------------------------------
def preprocess_image(img: Image.Image) -> Image.Image:
    """
    Pre-process the PIL image to improve OCR results.
    Converts to grayscale, enhances contrast, and applies a median filter.
    """
    gray = img.convert('L')
    enhancer = ImageEnhance.Contrast(gray)
    gray = enhancer.enhance(2)
    gray = gray.filter(ImageFilter.MedianFilter())
    return gray

def post_process_text(text: str) -> str:
    """
    Apply post-processing corrections to the OCR text.
    For example, replace common misinterpretations.
    """
    # Replace an uppercase 'O' with zero '0' in numeric contexts.
    text = re.sub(r'(?<=\d)O(?=\d)', '0', text)
    return text

# -------------------------------
# PDF and OCR Processing Functions
# -------------------------------
def extract_text(pdf_path: str) -> str:
    """
    Extract text from a PDF using PyPDF2; if that fails, fall back to OCR.
    Iterates page by page to reduce memory usage.
    Returns the extracted text.
    """
    text = ""
    try:
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            # Process one page at a time
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
            if len(text.strip()) > 100:
                return text
    except Exception as e:
        logging.error(f"Standard extraction failed: {e}")

    # If standard extraction fails, use OCR
    try:
        # Increase DPI for better resolution (300 DPI recommended)
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

def reorder_line(line: str, categories: list) -> str:
    """
    If a line contains one of the expected category keywords,
    remove it from its current position and prepend it.
    """
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
    """
    Clean the OCR text while preserving formatting.
    Keeps only lines longer than 10 characters that contain at least one digit
    or one of the expected category keywords; also removes URLs.
    Each kept line is reordered so that the category appears at the beginning.
    """
    cleaned_lines = []
    for line in raw_text.splitlines():
        line = re.sub(r"http\S+", "", line)  # Remove URLs
        if len(line) > 10 and (re.search(r'\d', line.lstrip()) or any(cat.lower() in line.lower() for cat in categories)):
            cleaned_lines.append(reorder_line(line, categories))
    return "\n".join(cleaned_lines)

def wrap_text_in_json(text: str) -> str:
    """
    Wrap the cleaned OCR text in a JSON structure that preserves formatting.
    Splits the text into lines and stores them in a "lines" array under "document".
    """
    lines = [line for line in text.splitlines() if line.strip() != ""]
    wrapped = {"document": {"lines": lines}}
    return json.dumps(wrapped, ensure_ascii=False, indent=2)

# -------------------------------
# Data Parsing Functions
# -------------------------------
def parse_value(value) -> float:
    """
    Convert Brazilian-formatted numbers (as strings) to float.
    Applies the SCALE_FACTOR.
    """
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
    """
    Extract a JSON block from the text.
    Searches for content enclosed in triple backticks; if not found,
    extracts from the first "{" to the last "}".
    """
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
    """
    Convert the API response into a structured JSON object.
    If the API returns a flat dictionary (with keys matching the categories),
    wrap it under "Ano Desconhecido".
    """
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
# API Integration Functions with Retry
# -------------------------------
def get_api_parameters(provider: str) -> tuple:
    """
    Returns API parameters (URL, headers, model, timeout) based on the chosen provider.
    Checks that necessary environment variables are set.
    """
    if provider == "deepseek":
        url = "http://localhost:11434/v1/chat/completions"
        headers = {}
        model = "deepseek-r1:14b"
        timeout_value = 240
    else:  # ChatGPT branch
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
    """
    Make an API call with the given parameters.
    This function is decorated with tenacity to retry on transient errors.
    """
    response = requests.post(url, headers=headers, json=payload, timeout=timeout_value)
    if response.status_code != 200:
        raise Exception(f"API call failed with status code {response.status_code}: {response.text}")
    return response.json()

def analyze_with_api(text_json: str, provider: str, categories: list) -> dict:
    """
    Analyze the provided OCR text (wrapped in JSON) using the chosen API.
    """
    prompt = f"""
Você é um especialista em contabilidade brasileira. A seguir, é fornecido um balanço patrimonial extraído por OCR, com a formatação preservada num objeto JSON.
Extraia somente os dados financeiros relevantes, convertendo "1.234,56" para 1234.56 e "(1.234,56)" para -1234.56; use NaN para valores ausentes.
Se houver dados de múltiplos anos, utilize os anos (4 dígitos) como chaves; caso contrário, utilize "Ano Desconhecido".
Retorne APENAS um objeto JSON com os valores extraídos.

Categorias:
{json.dumps(categories, indent=4, ensure_ascii=False)}

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
        return format_financial_data(response_json, categories)
    except Exception as e:
        logging.error(f"API Error: {e}")
        return None

def merge_analysis_results(results: list, categories: list) -> dict:
    """
    Merge multiple API responses (each a dict with year keys) into one JSON object.
    For each accounting year, non-NaN values are preferred.
    """
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

def analyze_document_in_batches(text_json: str, provider: str, categories: list, batch_size: int = 10000, overlap: int = 500) -> tuple:
    """
    Split the wrapped OCR JSON text into overlapping batches (by character count)
    and analyze each batch concurrently.
    Returns a tuple: (merged JSON analysis, batch processing logs)
    """
    logs = []
    try:
        doc_data = json.loads(text_json)
        if isinstance(doc_data.get("document"), dict) and "lines" in doc_data["document"]:
            doc = "\n".join(doc_data["document"]["lines"])
        else:
            doc = doc_data.get("document", "")
    except Exception as e:
        logs.append(f"Error loading wrapped JSON: {e}")
        return None, "\n".join(logs)

    batches = []
    start = 0
    while start < len(doc):
        end = start + batch_size
        batches.append(doc[start:end])
        start = end - overlap
    logs.append(f"Created {len(batches)} batches (batch_size={batch_size}, overlap={overlap}).")

    results = []
    total_batches = len(batches)
    # Dynamically set max_workers based on system capabilities.
    max_workers = min(32, (os.cpu_count() or 1) * 2)
    logs.append(f"Processing {total_batches} batches concurrently (max_workers={max_workers}).")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(analyze_with_api, wrap_text_in_json(batch), provider, categories): i+1
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
        merged_result = merge_analysis_results(results, categories)
        logs.append("Merged results from batches successfully.")
        return merged_result, "\n".join(logs)
    else:
        logs.append("No results were obtained from any batches.")
        return None, "\n".join(logs)

# -------------------------------
# Flask Web Application
# -------------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs("output", exist_ok=True)

UPLOAD_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Balance Sheet Analyzer</title>
  <style>
    body { font-family: Arial, sans-serif; background-color: #f2f2f2; margin: 0; padding: 0; }
    .container { width: 80%; margin: 50px auto; background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
    h2 { color: #333; }
    form { margin-top: 20px; }
    label { display: block; margin-bottom: 5px; }
    input[type="file"], select { padding: 10px; width: 100%; margin-bottom: 15px; }
    input[type="submit"] { padding: 10px 20px; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
    input[type="submit"]:hover { background: #45a049; }
    .progress { display: none; margin-top: 20px; text-align: center; }
    .progress p { font-size: 16px; color: #555; }
    .progress progress { width: 100%; height: 20px; }
  </style>
  <script>
    function showProgress() {
      document.getElementById("progressDiv").style.display = "block";
    }
  </script>
</head>
<body>
<div class="container">
  <h2>Balance Sheet Analyzer</h2>
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
  <div class="progress" id="progressDiv">
    <p>Processing... please wait.</p>
    <progress value="0" max="100" id="progressBar"></progress>
  </div>
</div>
</body>
</html>
"""

RESULT_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Analysis Result</title>
  <style>
    body { font-family: Arial, sans-serif; background-color: #f2f2f2; margin: 0; padding: 0; }
    .container { width: 80%; margin: 50px auto; background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); text-align: center; }
    a { display: block; margin: 15px 0; font-size: 18px; color: #4CAF50; text-decoration: none; }
    a:hover { text-decoration: underline; }
    pre { background: #f9f9f9; border: 1px solid #ddd; padding: 10px; text-align: left; overflow-x: auto; font-size: 14px; }
  </style>
</head>
<body>
<div class="container">
  <h2>Analysis Result for {{ filename }}</h2>
  {% if download_analysis_link %}
    <a href="{{ download_analysis_link }}">Download Analysis JSON File</a>
  {% else %}
    <p>No analysis result available.</p>
  {% endif %}
  <a href="{{ download_ocr_link }}">Download OCR JSON File</a>
  {% if download_xls_link %}
    <a href="{{ download_xls_link }}">Download Analysis XLS File</a>
  {% endif %}
  <h3>Batch Processing Logs</h3>
  <pre>{{ batch_logs }}</pre>
  <br>
  <a href="/">Upload another file</a>
</div>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(UPLOAD_HTML)

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
        result, batch_logs = analyze_document_in_batches(wrapped_json, provider, CATEGORIES, batch_size=10000, overlap=500)
    else:
        result = analyze_with_api(wrapped_json, provider, CATEGORIES)
        batch_logs = "Single API call used (no batch processing)."
    
    if result:
        analysis_json_path = os.path.join("output", f"{filename}_analysis.json")
        try:
            with open(analysis_json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=lambda x: "NaN" if math.isnan(x) else x)
        except Exception as e:
            logging.error(f"Error writing analysis JSON file: {e}")
        try:
            df = pd.DataFrame.from_dict(result, orient="index").T
            analysis_xls_path = os.path.join("output", f"{filename}_analysis.xlsx")
            df.to_excel(analysis_xls_path)
        except Exception as e:
            logging.error(f"Error writing analysis XLS file: {e}")
        
        download_analysis_link = url_for("download_file", filename=f"{filename}_analysis.json")
        download_ocr_link = url_for("download_file", filename=f"{filename}_ocr.json")
        download_xls_link = url_for("download_file", filename=f"{filename}_analysis.xlsx")
    else:
        download_analysis_link = None
        download_ocr_link = url_for("download_file", filename=f"{filename}_ocr.json")
        download_xls_link = None
        batch_logs += "\nNo analysis result obtained."
    
    return render_template_string(RESULT_HTML,
                                  filename=filename,
                                  download_analysis_link=download_analysis_link,
                                  download_ocr_link=download_ocr_link,
                                  download_xls_link=download_xls_link,
                                  batch_logs=batch_logs)

@app.route("/download/<path:filename>")
def download_file(filename: str):
    return send_from_directory("output", filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)