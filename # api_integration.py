# api_integration.py

import os
import json
import time
import logging
import requests
import concurrent.futures
from tenacity import retry, wait_exponential, stop_after_attempt
from data_parser import format_financial_data, merge_analysis_results
from ocr_utils import wrap_text_in_json

def get_api_parameters(provider: str) -> tuple:
    """
    Return API parameters (URL, headers, model, timeout) based on the chosen provider.
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
    Make an API call with exponential backoff for transient errors.
    """
    response = requests.post(url, headers=headers, json=payload, timeout=timeout_value)
    if response.status_code != 200:
        raise Exception(f"API call failed with status code {response.status_code}: {response.text}")
    return response.json()

def analyze_with_api(text_json: str, provider: str, categories: list) -> dict:
    """
    Analyze the provided OCR JSON using the chosen API.
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

def analyze_document_in_batches(text_json: str, provider: str, categories: list, batch_size: int = 10000, overlap: int = 500) -> tuple:
    """
    Split the OCR JSON text into overlapping batches and analyze them concurrently.
    Returns a tuple of (merged analysis, processing logs).
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