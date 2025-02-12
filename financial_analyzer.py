import os
import json
import math
import time
import requests
import re
from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract
from dotenv import load_dotenv
import concurrent.futures

# Load configurations
load_dotenv()

# Configuration – Mac specific
POPPLER_PATH = "/opt/homebrew/bin"  # Homebrew installation path
SCALE_FACTOR = int(os.getenv("SCALE_FACTOR", 1))
CATEGORIES_FILE = "categories.json"

def load_categories():
    """Load categories from JSON file."""
    with open(CATEGORIES_FILE, encoding="utf-8") as f:
        return json.load(f)["categories"]

def get_api_choice():
    """Let user choose API provider."""
    print("\nChoose AI provider:")
    print("1. DeepSeek API (recommended)")
    print("2. ChatGPT API")
    print("3. No AI analysis (text extraction only)")
    while True:
        choice = input("Enter 1, 2, or 3: ").strip()
        if choice in ["1", "2", "3"]:
            return {"1": "deepseek", "2": "chatgpt", "3": None}[choice]
        print("Invalid choice. Please try again.")

def extract_text(pdf_path):
    """
    Extract text from PDF using PyPDF2; if insufficient, fall back to OCR (optimized for Portuguese).
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
    Check if the line contains any of the expected category keywords.
    If so, remove the keyword from its current position and prepend it to the line.
    For example, if the line is:
      "(708.888,71) 3.2.4.01  ( 30931 )   DESPESAS FINANCEIRAS"
    it becomes:
      "DESPESAS FINANCEIRAS (708.888,71) 3.2.4.01  ( 30931 )"
    """
    found = None
    for cat in categories:
        # Use re.escape to match the literal category string; ignore case.
        if re.search(re.escape(cat), line, re.IGNORECASE):
            found = cat
            # Remove the keyword (all occurrences) from its current position.
            line = re.sub(re.escape(cat), '', line, flags=re.IGNORECASE)
            break  # If multiple categories exist, take the first found.
    if found:
        line = found + " " + line
    # Normalize whitespace and return.
    return re.sub(r'\s+', ' ', line).strip()

def clean_ocr_text(raw_text, categories):
    """
    Clean the OCR-extracted text while preserving its original formatting.
    A line is kept if it is longer than 10 characters and contains at least one digit
    or one of the expected category keywords.
    Also removes URLs.
    Each kept line is processed with reorder_line() so that the category appears at the beginning.
    """
    cleaned_lines = []
    for line in raw_text.splitlines():
        # Remove URLs.
        line = re.sub(r"http\S+", "", line)
        if len(line) > 10 and (re.search(r'\d', line.lstrip()) or any(cat.lower() in line.lower() for cat in categories)):
            reordered = reorder_line(line, categories)
            cleaned_lines.append(reordered)
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
    Convert Brazilian-formatted numbers (as strings) to float.
    
    - Removes "R$" and extra whitespace.
    - Normalizes minus signs (including en‑ and em‑dashes) and detects negatives (either with a leading "-" or if enclosed in parentheses).
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
    Extract a JSON block from the text.
    First, search for content enclosed in triple backticks; if none is found, extract from the first "{" to the last "}".
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

def analyze_with_api(text_json, provider, categories):
    """
    Analyze the provided OCR text (wrapped in JSON) using the chosen API.
    The prompt instructs the model (in Portuguese) to extract the balance sheet values
    from the structured document.
    """
    prompt = f"""
Você é um especialista em contabilidade brasileira. A seguir, é fornecido um balanço patrimonial extraído por OCR, com a formatação original preservada num objeto JSON.
Extraia somente os dados financeiros relevantes, convertendo números do formato "1.234,56" para 1234.56 e "(1.234,56)" para -1234.56; use NaN para valores ausentes.
Se houver dados de múltiplos anos, utilize os anos (4 dígitos) como chaves; caso contrário, utilize "Ano Desconhecido".
Retorne APENAS um objeto JSON com os valores extraídos.

Categorias:
{json.dumps(categories, indent=4, ensure_ascii=False)}

Documento (em JSON):
{text_json}
    """
    
    if provider == "deepseek":
        url = "https://api.deepseek.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {os.getenv('DEEPSEEK_API_KEY')}"}
        model = "deepseek-ai-default"
        timeout_value = 60
    else:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"}
        model = "gpt-4o-mini"
        timeout_value = 120

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
    For each accounting year, prefer non-NaN values.
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

def analyze_document_in_batches(text_json, provider, categories, batch_size=3000, overlap=500):
    """
    Split the document text (wrapped in JSON) into overlapping batches (by character count)
    and analyze each batch concurrently.
    Returns the merged JSON analysis.
    """
    try:
        doc_data = json.loads(text_json)
        # Join the lines back into a single string.
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
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
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

def main():
    os.makedirs("input", exist_ok=True)
    os.makedirs("output", exist_ok=True)
    
    print("=== Brazilian Balance Sheet Analyzer ===")
    print("Place PDFs in the 'input' folder first!\n")
    
    categories = load_categories()
    provider = get_api_choice()
    
    if provider == "chatgpt":
        batch_size = 3000
        overlap = 500
    else:
        batch_size = 3000
        overlap = 500
    
    for filename in os.listdir("input"):
        if not filename.lower().endswith(".pdf"):
            continue
            
        pdf_path = os.path.join("input", filename)
        print(f"\nProcessing: {filename}")
        
        raw_text = extract_text(pdf_path)
        cleaned_text = clean_ocr_text(raw_text, categories)
        print(f"Cleaned text length: {len(cleaned_text)} characters")
        print("Cleaned text snippet:")
        print(cleaned_text[:500])
        
        wrapped_json = wrap_text_in_json(cleaned_text)
        json_text_path = os.path.join("output", f"{filename}_ocr.json")
        with open(json_text_path, "w", encoding="utf-8") as f:
            f.write(wrapped_json)
        print(f"Saved wrapped OCR text to {json_text_path}")
        
        result = None
        if provider:
            print(f"Analyzing with {provider.capitalize()}...")
            if len(cleaned_text) > batch_size:
                result = analyze_document_in_batches(wrapped_json, provider, categories, batch_size=batch_size, overlap=overlap)
            else:
                result = analyze_with_api(wrapped_json, provider, categories)
            
            if result:
                analysis_json_path = os.path.join("output", f"{filename}_analysis.json")
                with open(analysis_json_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2, default=lambda x: "NaN" if math.isnan(x) else x)
                print(f"Saved analysis to {analysis_json_path}")
            else:
                print("No analysis result obtained.")
        
        print("=" * 50)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExecution interrupted by user. Exiting.")
    print("\nAll done! Check the 'output' folder for results.")