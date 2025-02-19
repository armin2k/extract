import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import math
import json
import logging
import time
import pandas as pd
from flask import Flask, request, render_template_string, send_from_directory, url_for, redirect, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
load_dotenv()

# Import our custom modules for processing and database
from pdf_processor import extract_text
from ocr_utils import clean_ocr_text, wrap_pages_in_json, extract_company_info_from_text
from api_integration import analyze_with_api, analyze_document_in_batches
from db import init_db, SessionLocal
from models import BalanceSheet
from tasks import process_balance_sheet  # Celery task

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

# HTML Templates
UPLOAD_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Balance Sheet Analyzer</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <!-- Bootstrap CSS -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    body { padding-top: 60px; }
  </style>
</head>
<body>
<nav class="navbar navbar-expand-lg navbar-light bg-light fixed-top">
  <div class="container-fluid">
    <a class="navbar-brand" href="/">Balance Sheet Analyzer</a>
    <div class="collapse navbar-collapse">
      <ul class="navbar-nav me-auto">
        <li class="nav-item"><a class="nav-link" href="/">Upload</a></li>
        <li class="nav-item"><a class="nav-link" href="/search">Search</a></li>
      </ul>
    </div>
  </div>
</nav>
<div class="container">
  <h2>Upload Balance Sheet</h2>
  <form method="post" enctype="multipart/form-data" action="/upload" onsubmit="showProgress();">
    <div class="mb-3">
      <label for="file" class="form-label">Select a Balance Sheet PDF:</label>
      <input type="file" class="form-control" name="file" id="file" required>
    </div>
    <div class="mb-3">
      <label for="provider" class="form-label">Select API Provider:</label>
      <select class="form-select" name="provider" id="provider">
        <option value="chatgpt">ChatGPT API (gpt-4o-mini)</option>
        <option value="deepseek">DeepSeek API (via Ollama offline)</option>
      </select>
    </div>
    <button type="submit" class="btn btn-primary">Upload</button>
  </form>
  <br>
  <div id="progressDiv" class="d-none">
    <div class="progress">
      <div id="progressBar" class="progress-bar" role="progressbar" style="width: 0%;">0%</div>
    </div>
    <p id="progressStatus" class="mt-2"></p>
    <p id="estimatedTime" class="mt-2"></p>
  </div>
</div>
<!-- Bootstrap JS -->
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
<script>
  function showProgress() {
    document.getElementById("progressDiv").classList.remove("d-none");
  }
</script>
</body>
</html>
"""

PROCESSING_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Processing Balance Sheet</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <!-- Bootstrap CSS -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
<nav class="navbar navbar-expand-lg navbar-light bg-light fixed-top">
  <div class="container-fluid">
    <a class="navbar-brand" href="/">Balance Sheet Analyzer</a>
    <div class="collapse navbar-collapse">
      <ul class="navbar-nav me-auto">
        <li class="nav-item"><a class="nav-link" href="/">Upload</a></li>
        <li class="nav-item"><a class="nav-link" href="/search">Search</a></li>
      </ul>
    </div>
  </div>
</nav>
<div class="container" style="padding-top: 80px;">
  <h2>Your balance sheet is being processed</h2>
  <p>Task ID: {{ task_id }}</p>
  <div class="progress">
    <div id="progressBar" class="progress-bar" role="progressbar" style="width: 0%;">0%</div>
  </div>
  <p id="progressStatus" class="mt-2"></p>
  <p id="estimatedTime" class="mt-2"></p>
  <script>
    function pollTaskStatus(taskId) {
      fetch("/task_status/" + taskId)
        .then(response => response.json())
        .then(data => {
          let progressBar = document.getElementById("progressBar");
          let progressStatus = document.getElementById("progressStatus");
          let estimatedTime = document.getElementById("estimatedTime");
          if (data.state === "PROGRESS") {
            let percent = Math.floor((data.meta.current / data.meta.total) * 100);
            progressBar.style.width = percent + "%";
            progressBar.textContent = percent + "%";
            progressStatus.textContent = data.meta.status;
            if(data.meta.eta) {
              estimatedTime.textContent = "Estimated time remaining: " + data.meta.eta + " seconds";
            }
            setTimeout(() => pollTaskStatus(taskId), 2000);
          } else if (data.state === "SUCCESS") {
            progressBar.style.width = "100%";
            progressBar.textContent = "100%";
            progressStatus.textContent = "Processing complete!";
            estimatedTime.textContent = "";
            window.location.href = "/view/" + data.meta.record_id;
          } else {
            progressStatus.textContent = "State: " + data.state;
            setTimeout(() => pollTaskStatus(taskId), 2000);
          }
        })
        .catch(error => {
          console.error("Error polling task status:", error);
          setTimeout(() => pollTaskStatus(taskId), 5000);
        });
    }
    pollTaskStatus("{{ task_id }}");
  </script>
</div>
<!-- Bootstrap JS -->
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

SEARCH_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Search Balance Sheets</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <!-- Bootstrap CSS -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
<div class="container">
  <nav class="navbar navbar-expand-lg navbar-light bg-light">
    <div class="container-fluid">
      <a class="navbar-brand" href="/">Balance Sheet Analyzer</a>
      <div class="collapse navbar-collapse">
        <ul class="navbar-nav me-auto">
          <li class="nav-item"><a class="nav-link" href="/">Upload</a></li>
          <li class="nav-item"><a class="nav-link" href="/search">Search</a></li>
        </ul>
      </div>
    </div>
  </nav>
  <h2 class="mt-4">Search Balance Sheets</h2>
  <form method="POST" action="/search">
    <div class="mb-3">
      <label for="company_name" class="form-label">Company Name:</label>
      <input type="text" name="company_name" id="company_name" class="form-control" placeholder="Enter company name">
    </div>
    <div class="mb-3">
      <label for="cnpj" class="form-label">CNPJ:</label>
      <input type="text" name="cnpj" id="cnpj" class="form-control" placeholder="Enter CNPJ">
    </div>
    <button type="submit" class="btn btn-primary">Search</button>
  </form>
  <br>
  <a href="/" class="btn btn-secondary">Back to Home</a>
</div>
<!-- Bootstrap JS -->
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

RESULTS_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Search Results</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <!-- Bootstrap CSS -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  <style>
    .action-link { color: blue; text-decoration: underline; cursor: pointer; }
  </style>
</head>
<body>
<div class="container">
  <nav class="navbar navbar-expand-lg navbar-light bg-light">
    <div class="container-fluid">
      <a class="navbar-brand" href="/">Balance Sheet Analyzer</a>
      <div class="collapse navbar-collapse">
        <ul class="navbar-nav me-auto">
          <li class="nav-item"><a class="nav-link" href="/">Upload</a></li>
          <li class="nav-item"><a class="nav-link" href="/search">Search</a></li>
        </ul>
      </div>
    </div>
  </nav>
  <h2 class="mt-4">Search Results for "{{ company_name }}" and "{{ cnpj }}"</h2>
  {% if results %}
    <table class="table table-striped">
      <thead>
        <tr>
          <th>ID</th>
          <th>Filename</th>
          <th>Company Name</th>
          <th>CNPJ</th>
          <th>Created At</th>
          <th>Action</th>
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
          <td><a class="action-link" href="{{ url_for('view_sheet', sheet_id=sheet.id) }}">View Details</a></td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  {% else %}
    <p>No results found.</p>
  {% endif %}
  <a href="/search" class="btn btn-secondary">Back to Search</a>
  <a href="/" class="btn btn-primary">Back to Home</a>
</div>
<!-- Bootstrap JS -->
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

VIEW_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Balance Sheet Detail</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <!-- Bootstrap CSS -->
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
<div class="container mt-4">
  <nav class="navbar navbar-expand-lg navbar-light bg-light">
    <div class="container-fluid">
      <a class="navbar-brand" href="/">Balance Sheet Analyzer</a>
      <div class="collapse navbar-collapse">
        <ul class="navbar-nav me-auto">
          <li class="nav-item"><a class="nav-link" href="/">Upload</a></li>
          <li class="nav-item"><a class="nav-link" href="/search">Search</a></li>
        </ul>
      </div>
    </div>
  </nav>
  <h2 class="mt-4">Balance Sheet Detail for {{ sheet.company_name }} ({{ sheet.cnpj }})</h2>
  <p><strong>Filename:</strong> {{ sheet.filename }}</p>
  <p><strong>Created At:</strong> {{ sheet.created_at }}</p>
  <h3>Extracted Data</h3>
  {% if data %}
    {% for year, details in data.items() %}
      <h4>{{ year }}</h4>
      <table class="table table-bordered">
        <thead>
          <tr>
            <th>Category</th>
            <th>Value</th>
          </tr>
        </thead>
        <tbody>
          {% for category, value in details.items() %}
            <tr>
              <td>{{ category }}</td>
              <td>{{ value }}</td>
            </tr>
          {% endfor %}
        </tbody>
      </table>
    {% endfor %}
  {% else %}
    <p>No extracted data available.</p>
  {% endif %}
  <a href="/search" class="btn btn-secondary">Back to Search</a>
  <a href="/" class="btn btn-primary">Back to Home</a>
</div>
<!-- Bootstrap JS -->
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

# ------------------------------
# Routes
# ------------------------------

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

        # Enqueue the processing task via Celery
        task = process_balance_sheet.delay(filename, upload_path, provider, CATEGORIES)
        return render_template_string(PROCESSING_HTML, task_id=task.id)
    except Exception as e:
        logger.exception("Error processing uploaded file:")
        return "Internal Server Error", 500

@app.route("/task_status/<task_id>")
def task_status(task_id):
    from celery.result import AsyncResult
    res = AsyncResult(task_id)
    response = {
        "state": res.state,
        "meta": res.info if res.info else {}
    }
    return jsonify(response)

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
        logger.info("Search requested with company_name: '%s' and cnpj: '%s'", company_name, cnpj)
        try:
            db = SessionLocal()
            query = db.query(BalanceSheet)
            if company_name:
                query = query.filter(BalanceSheet.company_name.ilike(f"%{company_name}%"))
            if cnpj:
                query = query.filter(BalanceSheet.cnpj.ilike(f"%{cnpj}%"))
            results = query.all()
            db.close()
            logger.info("Found %d results", len(results))
            return render_template_string(RESULTS_HTML, results=results, company_name=company_name, cnpj=cnpj)
        except Exception as e:
            logger.exception("Error during search query: %s", e)
            return f"Internal Server Error: {str(e)}", 500
    else:
        return render_template_string(SEARCH_HTML)

@app.route("/view/<int:sheet_id>", methods=["GET"])
def view_sheet(sheet_id: int):
    from db import SessionLocal
    from models import BalanceSheet
    db = SessionLocal()
    sheet = db.query(BalanceSheet).get(sheet_id)
    db.close()
    if sheet:
        try:
            data = json.loads(sheet.data)
        except Exception as e:
            logger.exception("Error parsing sheet data: %s", e)
            data = {}
        # Optionally, if your Celery task returns processing time in its meta, you can include it.
        processing_time = data.get("processing_time", "N/A")
        return render_template_string(VIEW_HTML, sheet=sheet, data=data, processing_time=processing_time)
    else:
        return "Record not found", 404

if __name__ == "__main__":
    init_db()  # Initialize the database and create tables if they don't exist
    app.debug = True  # Enable debug mode for development (disable in production)
    app.run(host="0.0.0.0", port=8000)