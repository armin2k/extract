# data_parser.py

import math
import json
import re
import logging

def parse_value(value) -> float:
    """
    Convert Brazilian-formatted numbers (e.g., "1.234,56" or "(1.234,56)") into floats.
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
        SCALE_FACTOR = 1  # Adjust if needed
        return num * SCALE_FACTOR
    try:
        SCALE_FACTOR = 1
        return float(value) * SCALE_FACTOR
    except Exception:
        return math.nan

def extract_json_from_text(text: str) -> str:
    """
    Extract a JSON block from the text.
    First, search for content enclosed in triple backticks; if not found,
    extract from the first '{' to the last '}'.
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
    If the response is a flat dictionary matching your categories,
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

def merge_analysis_results(results: list, categories: list) -> dict:
    """
    Merge multiple API responses (each a dict with year keys) into one JSON object.
    For each year, non-NaN values are preferred.
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