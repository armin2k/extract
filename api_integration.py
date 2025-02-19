# api_integration.py
import os
import json
import logging
import requests
from tenacity import retry, wait_exponential, stop_after_attempt
from data_parser import format_financial_data, merge_analysis_results
from ocr_utils import wrap_pages_in_json  # Ensure this import is present

logger = logging.getLogger(__name__)

# Create a persistent session for API calls.
session = requests.Session()

def get_api_parameters(provider: str):
    """
    Return the API endpoint URL, headers, model, and timeout value based on the provider.
    """
    if provider == "deepseek":
        url = "http://localhost:11434/v1/chat/completions"
        headers = {}
        model = "deepseek-r1:14b"
        timeout_value = 240
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}"}
        model = "gpt-4o-mini"
        timeout_value = 180
    return url, headers, model, timeout_value

@retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
def make_api_call(url: str, headers: dict, payload: dict, timeout_value: int):
    """
    Make an API call with retries using exponential backoff.
    """
    response = session.post(url, headers=headers, json=payload, timeout=timeout_value)
    if response.status_code != 200:
        raise Exception(f"API call failed with status {response.status_code}: {response.text}")
    return response.json()

def analyze_with_api(text_json: str, provider: str, categories: list):
    """
    Analyze the provided OCR JSON using the chosen API.
    Constructs a prompt using the categories and text_json, makes the API call, and formats the response.
    """
    prompt = f"""
Você é um especialista em contabilidade brasileira. Extraia os dados financeiros relevantes a partir do documento JSON a seguir.
Retorne somente um objeto JSON com os valores extraídos.

Categorias:
{json.dumps(categories, indent=4, ensure_ascii=False)}

Documento (em JSON):
{text_json}
    """
    url, headers, model, timeout_value = get_api_parameters(provider)
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.1}
    try:
        response_json = make_api_call(url, headers, payload, timeout_value)
        return format_financial_data(response_json, categories)
    except Exception as e:
        logger.exception("Error during API call: %s", e)
        return {}

def analyze_document_in_batches(text_json: str, provider: str, categories: list, batch_size: int = 10000, overlap: int = 500):
    """
    Splits the provided JSON text into overlapping batches and processes them concurrently via API calls.
    Merges the results and returns the final JSON along with processing logs.
    """
    logs = []
    try:
        doc_data = json.loads(text_json)
        if isinstance(doc_data.get("document"), dict) and "pages" in doc_data["document"]:
            doc = "\n".join(["\n".join(page.get("lines", [])) for page in doc_data["document"]["pages"]])
        else:
            doc = doc_data.get("document", "")
    except Exception as e:
        logs.append(f"Error parsing JSON: {e}")
        return {}, "\n".join(logs)
    
    # Create overlapping batches.
    batches = []
    start = 0
    while start < len(doc):
        end = start + batch_size
        batches.append(doc[start:end])
        start = end - overlap
    logs.append(f"Created {len(batches)} batches (batch_size={batch_size}, overlap={overlap}).")
    
    # Process batches concurrently.
    import concurrent.futures
    results = []
    max_workers = min(32, (os.cpu_count() or 1) * 2)
    logs.append(f"Processing {len(batches)} batches concurrently with {max_workers} workers.")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_batch = {
            executor.submit(analyze_with_api, wrap_pages_in_json([batch]), provider, categories): idx
            for idx, batch in enumerate(batches)
        }
        for future in concurrent.futures.as_completed(future_to_batch):
            idx = future_to_batch[future]
            try:
                batch_result = future.result()
                if batch_result:
                    results.append(batch_result)
                    logs.append(f"Batch {idx} processed successfully.")
                else:
                    logs.append(f"Batch {idx} returned no result.")
            except Exception as e:
                logs.append(f"Batch {idx} raised an exception: {e}")
    if results:
        merged_result = merge_analysis_results(results, categories)
        logs.append("Merged results from batches successfully.")
        return merged_result, "\n".join(logs)
    else:
        logs.append("No results were obtained from any batches.")
        return {}, "\n".join(logs)