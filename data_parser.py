# data_parser.py
import json
import math
import logging
from ocr_utils import extract_json_from_text  # Ensure this function is defined in ocr_utils.py

logger = logging.getLogger(__name__)
SCALE_FACTOR = 1  # Adjust as needed

def parse_value(value):
    """
    Convert a Brazilian-formatted number (as a string or number) to a float.
    If the value is a dict:
      - If it has exactly one key, use that value.
      - If it has multiple keys and the keys represent years (numeric), choose the value corresponding to the highest year.
      - Otherwise, fall back to the first key's value.
    """
    if isinstance(value, dict):
        try:
            # Try to convert keys to integers (assuming they are years)
            year_keys = {int(k): v for k, v in value.items() if k.isdigit() or k.isnumeric()}
            if year_keys:
                # Choose the value for the maximum year
                max_year = max(year_keys.keys())
                value = year_keys[max_year]
            else:
                # If not all keys are numeric, simply pick the first key's value
                value = next(iter(value.values()))
        except Exception as e:
            logger.warning("Error processing dict value %s: %s. Using first value.", value, e)
            value = next(iter(value.values()))
    if value in [None, "", "NaN"]:
        return math.nan
    try:
        if isinstance(value, str):
            # Remove currency symbols and spaces
            value = value.replace("R$", "").strip()
            # Remove thousand separators and replace decimal comma with dot.
            value = value.replace(".", "").replace(",", ".")
        return float(value) * SCALE_FACTOR
    except Exception as e:
        logger.exception("Error converting value %s: %s", value, e)
        return math.nan

def format_financial_data(response_json, categories):
    """
    Convert the API response into a structured JSON object.
    - Extract JSON text from the response content.
    - If the returned raw data is flat and keys match categories, wrap it under "Ano Desconhecido".
    - For each year and category, parse the value using parse_value.
    This version handles both simple numbers and dictionaries.
    """
    try:
        content = response_json.get('choices', [{}])[0].get('message', {}).get('content', '')
        if not content.strip():
            logger.error("Empty content in API response.")
            return None
        json_text = extract_json_from_text(content)
        if not json_text:
            logger.error("Could not extract JSON from API response.")
            return None
        raw_data = json.loads(json_text)
        # If raw_data is flat (keys matching categories), wrap it under a default key.
        if any(key in categories for key in raw_data.keys()):
            raw_data = {"Ano Desconhecido": raw_data}
        formatted = {}
        for year, data in raw_data.items():
            formatted[year] = {}
            # Check if data is a dictionary; if not, treat it as a single value for each category.
            for category in categories:
                if isinstance(data, dict):
                    raw_value = data.get(category, math.nan)
                else:
                    raw_value = data
                formatted[year][category] = parse_value(raw_value)
        return formatted
    except Exception as e:
        logger.exception("Error formatting financial data: %s", e)
        return None

def merge_analysis_results(results, categories):
    """
    Merge multiple analysis result dictionaries.
    For each year, use non-NaN values from later results if available.
    """
    merged = {}
    for result in results:
        if not result:
            continue
        for year, data in result.items():
            if year not in merged:
                merged[year] = data.copy()
            else:
                for category in categories:
                    current_val = merged[year].get(category, math.nan)
                    new_val = data.get(category, math.nan)
                    if math.isnan(current_val) and not math.isnan(new_val):
                        merged[year][category] = new_val
    return merged

def extract_json_from_text(text):
    """
    Attempt to extract a JSON block from the text.
    Looks first for content enclosed in triple backticks (optionally with "json"),
    then falls back to extracting from the first '{' to the last '}'.
    Returns the extracted JSON string if successful, or None if not found.
    """
    import re
    import json

    # Look for triple backticks encapsulating JSON
    candidates = re.findall(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    for candidate in candidates:
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            continue

    # Fallback: extract from first '{' to last '}'
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