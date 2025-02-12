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
from flask import Flask, request, render_template_string, send_from_directory, url_for

# Load configurations from .env file
load_dotenv()

# Configuration – Mac specific (adjust as needed)
POPPLER_PATH = os.getenv("POPPLER_PATH", "/opt/homebrew/bin")
SCALE_FACTOR = int(os.getenv("SCALE_FACTOR", 1))
CATEGORIES_FILE = "categories.json"

def load_categories():
    """Load categories from JSON file."""
    with open(CATEGORIES_FILE, encoding="utf-8") as f:
        return json.load(f)["categories"]

CATEGORIES = load_categories()

def get_api_choice():
    """(For console use) Let user choose API provider."""
    print("\nChoose API provider:")
    print("1. DeepSeek API (Ollama offline if OLLAMA_MODE=offline)")
    print("2. ChatGPT API")
    print("3. No AI analysis (text extraction only)")
    while True:
        choice = input("Enter 1, 2, or 3: ").strip()
        if choice in ["1", "2", "3"]:
            return {"1": "deepseek", "2": "chatgpt", "3": None}[choice]
        print("Invalid choice. Please try again.")

def extract_text(pdf_path):
    """
    Extract text from a PDF using PyPDF2; if that fails, use OCR (optimized for Portuguese).
    Returns the extracted text.
    """
    try:
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            text = "\n".join([page.extract_text() or "" for page in reader.pages])
            if len(text.strip()) > 100:
                return text
    except Exception as e:
        print(f"Standard extraction failed: {str(e)}")
    try:
        images = convert_from_path(pdf_path, poppler_path=POPPLER_PATH)
        ocr_text = []
        custom_config = "--psm 6 --oem 3"
        for img in images:
            text_img = pytesseract.image_to_string(img, lang='por', config=custom_config)
            ocr_text.append(text_img)
        return "\n".join(ocr_text)
    except Exception as e:
        print(f"OCR failed: {str(e)}")
        return ""

def reorder_line(line, categories):
    """
    If a line contains an expected category keyword, remove it from its current position and prepend it.
    """
    found = None
    for cat in categories:
        if re.search(re.escape(cat), line, re.IGNORECASE):
            found = cat
            line = re.sub(re.escape(cat), '', line, flags=re.IGNORECASE)
            break
    if found:
        line = found + " " + line
    return re.sub(r'\s+', ' ', line).strip()

def clean_ocr_text(raw_text, categories):
    """
    Clean the OCR-extracted text while preserving its original formatting.
    Keeps only lines longer than 10 characters that contain at least one digit
    or one of the expected category keywords; also removes URLs.
    Each kept line is reordered so that the category appears at the beginning.
    """
    cleaned_lines = []
    for line in raw_text.splitlines():
        # Remove URLs
        line = re.sub(r"http\S+", "", line)
        if len(line) > 10 and (re.search(r'\d', line.lstrip()) or any(cat.lower() in line.lower() for cat in categories)):
            cleaned_lines.append(reorder_line(line, categories))
    return "\n".join(cleaned_lines)

def wrap_text_in_json(text):
    """
    Wrap the cleaned OCR text in a JSON structure that preserves formatting.
    The text is split into lines (preserving indents) and stored under the key "document" with a "lines" array.
    """
    lines = [line for line in text.splitlines() if line.strip() != ""]
    wrapped = {"document": {"lines": lines}}
    return json.dumps(wrapped, ensure_ascii=False, indent=2)

def parse_value(value):
    """
    Convert Brazilian-formatted numbers (as strings) to a float.
    - Removes "R$" and extra whitespace.
    - Normalizes minus signs (including en‑/em‑dashes) and detects negatives via a leading "-" or enclosed in parentheses.
    - Removes thousand separators and converts the decimal comma to a dot.
    - Applies SCALE_FACTOR.
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
    except:
        return math.nan

def extract_json_from_text(text):
    """
    Extract a JSON block from text.
    First, search for content enclosed in triple backticks; if not found, extract from the first "{" to the last "}".
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
            return None
    return None

def format_financial_data(response_json, categories):
    """
    Convert the API response into a structured JSON object.
    If the API returns a flat dictionary (with keys matching the categories), wrap it under "Ano Desconhecido".
    """
    try:
        content = response_json.get('choices', [{}])[0].get('message', {}).get('content', '')
        if not content.strip():
            print("Formatting error: API returned empty content.")
            return None
        json_text = extract_json_from_text(content)
        if not json_text:
            print("Formatting error: Could not extract JSON content from the response.")
            print("Raw response snippet:")
            print(content[:500])
            return None
        try:
            raw_data = json.loads(json_text)
        except json.JSONDecodeError as jde:
            print(f"Formatting error: Failed to decode JSON.\nExtracted content:\n{json_text}\nError: {jde}")
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
        print(f"Formatting error: {str(e)}")
        return None

def get_api_parameters(provider):
    """
    Return the API endpoint, headers, model, and timeout based on the chosen provider.
    For DeepSeek, if OLLAMA_MODE=offline is set in the environment, use the local Ollama endpoint.
    """
    if provider == "deepseek":
        if os.getenv("OLLAMA_MODE", "online").lower() == "offline":
            url = "http://localhost:11434/v1/chat/completions"  # Local Ollama endpoint
            headers = {}  # No auth needed locally
        else:
            url = "https://api.deepseek.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}"}
        model = "deepseek-r1:7b"  # Use your specific DeepSeek model
        timeout_value = 120
    else:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}
        model = "gpt-4o-mini"
        timeout_value = 120
    return url, headers, model, timeout_value

def analyze_with_api(text_json, provider, categories):
    """
    Analyze the provided OCR text (wrapped in JSON) using the chosen API.
    The prompt instructs the model to extract the financial data from the structured document.
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
    retries = 3
    for attempt in range(retries):
        try:
            response = requests.post(
                url,
                headers=headers,
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.1
                },
                timeout=timeout_value
            )
            response_json = response.json()
            return format_financial_data(response_json, categories)
        except requests.exceptions.Timeout:
            if attempt < retries - 1:
                print(f"Timeout - Retrying ({attempt+1}/{retries})...")
                time.sleep(5)
                continue
            else:
                print("API timeout after 3 attempts")
                return None
        except Exception as e:
            print(f"API Error: {str(e)}")
            return None

def merge_analysis_results(results, categories):
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

def analyze_document_in_batches(text_json, provider, categories, batch_size=10000, overlap=500):
    """
    Split the document text (wrapped in JSON) into overlapping batches (by character count)
    and analyze each batch concurrently.
    Returns the merged JSON analysis.
    """
    try:
        doc_data = json.loads(text_json)
        if isinstance(doc_data.get("document"), dict) and "lines" in doc_data["document"]:
            doc = "\n".join(doc_data["document"]["lines"])
        else:
            doc = doc_data.get("document", "")
    except Exception as e:
        print(f"Error loading wrapped JSON: {e}")
        return None

    batches = []
    start = 0
    while start < len(doc):
        end = start + batch_size
        batches.append(doc[start:end])
        start = end - overlap

    results = []
    total_batches = len(batches)
    print(f"Processing {total_batches} batches concurrently...")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
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
                else:
                    print(f"Warning: No result from batch {i}.")
            except Exception as exc:
                print(f"Batch {i} generated an exception: {exc}")
    if results:
        merged_result = merge_analysis_results(results, categories)
        return merged_result
    else:
        return None

# --- Flask Web Application ---

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# HTML template for uploading files with improved layout and a progress indicator
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
    input[type="file"], select { padding: 10px; width: 100%; margin-bottom: 15px; }
    input[type="submit"] { padding: 10px 20px; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
    input[type="submit"]:hover { background: #45a049; }
    .progress { display: none; margin-top: 20px; text-align: center; }
    .progress p { font-size: 16px; color: #555; }
    .progress progress { width: 100%; height: 20px; }
  </style>
</head>
<body>
<div class="container">
  <h2>Balance Sheet Analyzer</h2>
  <form method="post" enctype="multipart/form-data" action="/upload" onsubmit="showProgress()">
    <label for="file">Select a Balance Sheet PDF:</label>
    <input type="file" name="file" id="file">
    <label for="provider">Select API Provider:</label>
    <select name="provider" id="provider">
      <option value="chatgpt">ChatGPT API</option>
      <option value="deepseek">DeepSeek API (Ollama offline if OLLAMA_MODE=offline)</option>
    </select>
    <input type="submit" value="Upload">
  </form>
  <div class="progress" id="progressDiv">
    <p>Processing... please wait.</p>
    <progress value="0" max="100" id="progressBar"></progress>
  </div>
</div>
<script>
  function showProgress() {
    document.getElementById("progressDiv").style.display = "block";
  }
</script>
</body>
</html>
"""

# HTML template for displaying the result with download links (without displaying the JSON text)
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
    provider = request.form.get("provider", "chatgpt")
    filename = file.filename
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(upload_path)
    
    # Process the PDF
    raw_text = extract_text(upload_path)
    cleaned_text = clean_ocr_text(raw_text, CATEGORIES)
    wrapped_json = wrap_text_in_json(cleaned_text)
    
    os.makedirs("output", exist_ok=True)
    json_text_path = os.path.join("output", f"{filename}_ocr.json")
    with open(json_text_path, "w", encoding="utf-8") as f:
        f.write(wrapped_json)
    
    # Use batch processing if text is long
    if len(cleaned_text) > 2000:
        result = analyze_document_in_batches(wrapped_json, provider, CATEGORIES, batch_size=2000, overlap=500)
    else:
        result = analyze_with_api(wrapped_json, provider, CATEGORIES)
    
    # Save analysis result files
    if result:
        analysis_json_path = os.path.join("output", f"{filename}_analysis.json")
        with open(analysis_json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False, default=lambda x: "NaN" if math.isnan(x) else x)
        # Convert the result JSON to an Excel file with years as columns and categories as rows.
        # We first create a DataFrame with years as index, then transpose it.
        df = pd.DataFrame.from_dict(result, orient="index").T
        analysis_xls_path = os.path.join("output", f"{filename}_analysis.xlsx")
        df.to_excel(analysis_xls_path)
        
        download_analysis_link = url_for("download_file", filename=f"{filename}_analysis.json")
        download_ocr_link = url_for("download_file", filename=f"{filename}_ocr.json")
        download_xls_link = url_for("download_file", filename=f"{filename}_analysis.xlsx")
    else:
        download_analysis_link = None
        download_ocr_link = url_for("download_file", filename=f"{filename}_ocr.json")
        download_xls_link = None
    
    return render_template_string(RESULT_HTML,
                                  filename=filename,
                                  download_analysis_link=download_analysis_link,
                                  download_ocr_link=download_ocr_link,
                                  download_xls_link=download_xls_link)

@app.route("/download/<path:filename>")
def download_file(filename):
    return send_from_directory("output", filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)