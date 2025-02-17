# app.py

import os
import json
import math
import logging
import pandas as pd
from flask import Flask, request, render_template_string, send_from_directory, url_for
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Import our modules
from pdf_processor import extract_text
from ocr_utils import clean_ocr_text, wrap_pages_in_json
from api_integration import analyze_with_api, analyze_document_in_batches

# Load environment variables and set up logging
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Configuration
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("output", exist_ok=True)
CATEGORIES_FILE = "categories.json"





def load_categories():
    try:
        with open(CATEGORIES_FILE, encoding="utf-8") as f:
            data = json.load(f)
            return data["categories"]
    except Exception as e:
        logging.error(f"Failed to load categories: {e}")
        return []

CATEGORIES = load_categories()

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

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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

    # Process the PDF into a list of page texts.
    raw_pages = extract_text(upload_path)
    # Clean each page separately.
    cleaned_pages = [clean_ocr_text(page, CATEGORIES) for page in raw_pages]
    # Wrap the cleaned pages into a structured JSON.
    wrapped_json = wrap_pages_in_json(cleaned_pages)
    
    # Save OCR output for download.
    json_text_path = os.path.join("output", f"{filename}_ocr.json")
    try:
        with open(json_text_path, "w", encoding="utf-8") as f:
            f.write(wrapped_json)
    except Exception as e:
        logging.error(f"Error writing OCR JSON file: {e}")

    # Decide whether to use batch processing for API analysis.
    total_text_length = sum(len(page) for page in cleaned_pages)
    if total_text_length > 10000:
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
