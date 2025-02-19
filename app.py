import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import math
import json
import logging
import pandas as pd
from flask import Flask, request, render_template_string, send_from_directory, url_for
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Import our custom modules for processing and database
from pdf_processor import extract_text
from ocr_utils import clean_ocr_text, wrap_pages_in_json
from api_integration import analyze_with_api, analyze_document_in_batches
from db import init_db, SessionLocal
from models import BalanceSheet

# Load environment variables and configure logging
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs("output", exist_ok=True)

CATEGORIES_FILE = "categories.json"

def load_categories() -> list:
    """Load balance sheet categories from a JSON file."""
    try:
        with open(CATEGORIES_FILE, encoding="utf-8") as f:
            data = json.load(f)
            return data.get("categories", [])
    except Exception as e:
        logger.exception("Failed to load categories: %s", e)
        return []

CATEGORIES = load_categories()

# HTML templates

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
    .menu a { margin-right: 15px; }
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
  <div class="menu">
    <a href="/">Home</a> |
    <a href="/search">Search Balance Sheets</a>
  </div>
  <h2>Upload Balance Sheet</h2>
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
  <div class="menu">
    <a href="/">Home</a> |
    <a href="/search">Search Balance Sheets</a>
  </div>
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

SEARCH_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Search Balance Sheets</title>
  <style>
    body { font-family: Arial, sans-serif; background-color: #f2f2f2; margin: 0; padding: 0; }
    .container { width: 80%; margin: 50px auto; background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
    input[type="text"] { width: 100%; padding: 10px; margin: 10px 0; }
    input[type="submit"] { padding: 10px 20px; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; }
    input[type="submit"]:hover { background: #45a049; }
    .menu a { margin-right: 15px; }
  </style>
</head>
<body>
<div class="container">
  <div class="menu">
    <a href="/">Home</a> |
    <a href="/search">Search Balance Sheets</a>
  </div>
  <h2>Search Balance Sheets</h2>
  <form method="POST" action="/search">
    <label for="company_name">Company Name:</label>
    <input type="text" name="company_name" id="company_name" placeholder="Enter company name">
    <label for="cnpj">CNPJ:</label>
    <input type="text" name="cnpj" id="cnpj" placeholder="Enter CNPJ">
    <input type="submit" value="Search">
  </form>
  <br>
  <a href="/">Back to Home</a>
</div>
</body>
</html>
"""

RESULTS_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Search Results</title>
  <style>
    body { font-family: Arial, sans-serif; background-color: #f2f2f2; margin: 0; padding: 0; }
    .container { width: 80%; margin: 50px auto; background: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }
    table { width: 100%; border-collapse: collapse; }
    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
    th { background-color: #4CAF50; color: white; }
    .menu a { margin-right: 15px; }
  </style>
</head>
<body>
<div class="container">
  <div class="menu">
    <a href="/">Home</a> |
    <a href="/search">Search Balance Sheets</a>
  </div>
  <h2>Search Results for "{{ company_name }}" and "{{ cnpj }}"</h2>
  {% if results %}
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Filename</th>
          <th>Company Name</th>
          <th>CNPJ</th>
          <th>Created At</th>
        </tr>
      </thead>
      <tbody>
        {% for sheet in results %}
        <tr>
          <td>{{ sheet.id }}</td>
          <td>{{ sheet.filename }}</td>
          <td>{{ sheet.company_name }}</td>
          <td>{{ sheet.cnpj }}</td>
          <td>{{ sheet.created_at }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  {% else %}
    <p>No results found.</p>
  {% endif %}
  <br>
  <a href="/search">Back to Search</a>
  <br>
  <a href="/">Back to Home</a>
</div>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def index():
    return render_template_string(UPLOAD_HTML)

@app.route("/upload", methods=["POST"])
def upload():
    try:
        # 1. Validate and save the uploaded file
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
        logger.info(f"File {filename} saved to {upload_path}.")

        # 2. Process the PDF: extract pages, clean them, and wrap into JSON
        raw_pages = extract_text(upload_path)
        cleaned_pages = [clean_ocr_text(page, CATEGORIES) for page in raw_pages]
        wrapped_json = wrap_pages_in_json(cleaned_pages)
        
        # Save OCR JSON output for download
        json_text_path = os.path.join("output", f"{filename}_ocr.json")
        with open(json_text_path, "w", encoding="utf-8") as f:
            f.write(wrapped_json)
        
        # 3. Analyze the document using the API
        total_text_length = sum(len(page) for page in cleaned_pages)
        if total_text_length > 10000:
            result, batch_logs = analyze_document_in_batches(wrapped_json, provider, CATEGORIES, batch_size=10000, overlap=500)
        else:
            result = analyze_with_api(wrapped_json, provider, CATEGORIES)
            batch_logs = "Single API call used (no batch processing)."
        
        # 4. Save the analysis result to the database (if available)
        if result:
            db = SessionLocal()  # Open a database session
            new_sheet = BalanceSheet(
                filename=filename,
                data=json.dumps(result)
                # If you have a way to extract company name and CNPJ, add them here:
                # company_name="Extracted Company Name",
                # cnpj="Extracted CNPJ"
            )
            db.add(new_sheet)
            db.commit()
            db.close()

            # Save analysis result files for download
            analysis_json_path = os.path.join("output", f"{filename}_analysis.json")
            with open(analysis_json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False, default=lambda x: "NaN" if math.isnan(x) else x)
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
            batch_logs += "\nNo analysis result obtained."
        
        return render_template_string(RESULT_HTML,
                                      filename=filename,
                                      download_analysis_link=download_analysis_link,
                                      download_ocr_link=download_ocr_link,
                                      download_xls_link=download_xls_link,
                                      batch_logs=batch_logs)
    except Exception as e:
        logger.exception("Error processing uploaded file:")
        return "Internal Server Error", 500

@app.route("/download/<path:filename>")
def download_file(filename: str):
    return send_from_directory("output", filename, as_attachment=True)

@app.route("/search", methods=["GET", "POST"])
def search():
    from db import SessionLocal
    from models import BalanceSheet
    if request.method == "POST":
        company_name = request.form.get("company_name", "").strip()
        cnpj = request.form.get("cnpj", "").strip()
        db = SessionLocal()
        query = db.query(BalanceSheet)
        if company_name:
            query = query.filter(BalanceSheet.company_name.ilike(f"%{company_name}%"))
        if cnpj:
            query = query.filter(BalanceSheet.cnpj.ilike(f"%{cnpj}%"))
        results = query.all()
        db.close()
        return render_template_string(RESULTS_HTML, results=results, company_name=company_name, cnpj=cnpj)
    else:
        return render_template_string(SEARCH_HTML)

if __name__ == "__main__":
    init_db()  # Initialize the database and create tables if they don't exist
    app.run(host="0.0.0.0", port=8000)