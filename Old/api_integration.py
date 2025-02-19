import os
import json
import time
import logging
import requests
from data_parser import format_financial_data, merge_analysis_results

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def get_api_parameters(provider: str) -> tuple:
    if provider == "deepseek":
        url = "http://localhost:11434/v1/chat/completions"
        headers = {}
        model = "deepseek-r1:14b"
        timeout_value = 240
    else:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            logging.error("OPENAI_API_KEY is not set.")
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}"}
        model = "gpt-4o-mini"
        timeout_value = 180
    return url, headers, model, timeout_value

def analyze_with_api(text_json: str, provider: str, categories: list) -> dict:
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
            if response.status_code != 200:
                logging.error(f"API call failed with status {response.status_code}: {response.text}")
                time.sleep(5)
                continue
            response_json = response.json()
            return format_financial_data(response_json, categories)
        except requests.exceptions.Timeout:
            logging.warning(f"Timeout - Retrying ({attempt+1}/{retries})...")
            time.sleep(5)
            continue
        except Exception as e:
            logging.error(f"API Error: {e}")
            return None
    return None

def analyze_document_in_batches(text_json: str, provider: str, categories: list, batch_size: int = 10000, overlap: int = 500) -> tuple:
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
    logs.append(f"Processing {total_batches} batches concurrently (max_workers=3).")
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_index = {
            executor.submit(analyze_with_api, wrap_text_in_json(batch), provider, categories): i+1
            for i, batch in enumerate(batches)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            i = future_to_index[future]
            try:
                res = future.result()
                if res:
                    results.append(res)
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