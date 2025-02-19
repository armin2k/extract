# data_parser.py
import json
import math
import logging
from ocr_utils import extract_json_from_text  # Ensure this function is defined in ocr_utils.py

logger = logging.getLogger(__name__)
SCALE_FACTOR = 1  # Adjust if needed

def parse_value(value):
    """
    Convert a Brazilian-formatted number (as a string or number) to a float.
    If the value is None, empty, or "NaN", returns math.nan.
    If the value is a string, it removes currency symbols, thousand separators,
    and converts a decimal comma to a dot before converting to float.
    If value is already an int or float, it returns it multiplied by SCALE_FACTOR.
    """
    if value in [None, "", "NaN"]:
        return math.nan
    if isinstance(value, (int, float)):
        return float(value) * SCALE_FACTOR
    try:
        # Assume it's a string: remove common currency symbols and spaces
        value = value.replace("R$", "").strip()
        # Remove thousand separators (.) and replace the decimal comma (,) with dot (.)
        value = value.replace(".", "").replace(",", ".")
        return float(value) * SCALE_FACTOR
    except Exception as e:
        logger.exception("Error converting value %s: %s", value, e)
        return math.nan

def format_financial_data(response_json, categories):
    """
    Convert the API response into a structured JSON object.
    
    Steps:
      1. Extract a JSON block from the API response.
      2. If the raw data is flat (its keys do not all represent years), wrap it under "Ano Desconhecido".
         Otherwise, preserve the multi-year structure.
      3. For each year and for each category, if the value is:
           - A simple value: convert it using parse_value.
           - A dict (e.g. multi-year sub-values): convert each sub-value.
    """
    try:
        # Extract the text content from the API response.
        content = response_json.get('choices', [{}])[0].get('message', {}).get('content', '')
        if not content.strip():
            logger.error("Empty content in API response.")
            return None

        json_text = extract_json_from_text(content)
        if not json_text:
            logger.error("Could not extract JSON from API response.")
            return None

        raw_data = json.loads(json_text)

        # Determine if raw_data's keys represent years (e.g. "2020", "2021", etc.)
        if isinstance(raw_data, dict) and all(isinstance(k, str) and k.strip().isdigit() for k in raw_data.keys()):
            multi_year = True
        else:
            multi_year = False
            raw_data = {"Ano Desconhecido": raw_data}

        formatted = {}
        for year, data in raw_data.items():
            formatted[year] = {}
            for category in categories:
                # If data is a dictionary, get the value for the category.
                # Otherwise, assume data itself is the value.
                raw_value = data.get(category, None) if isinstance(data, dict) else data
                # If the raw value is a dictionary, process each inner key/value.
                if isinstance(raw_value, dict):
                    inner_converted = {}
                    for subkey, subval in raw_value.items():
                        inner_converted[subkey] = parse_value(subval)
                    formatted[year][category] = inner_converted
                else:
                    formatted[year][category] = parse_value(raw_value)
        return formatted
    except Exception as e:
        logger.exception("Error formatting financial data: %s", e)
        return None

def merge_analysis_results(results, categories):
    """
    Merge multiple analysis result dictionaries.
    For each year and category, if the merged value is NaN and a new result provides a non-NaN value,
    update the merged data.
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
                    if not isinstance(current_val, dict) and math.isnan(current_val) and not math.isnan(new_val):
                        merged[year][category] = new_val
                    elif isinstance(current_val, dict) and isinstance(new_val, dict):
                        for subkey, subval in new_val.items():
                            if math.isnan(current_val.get(subkey, math.nan)) and not math.isnan(subval):
                                merged[year][category][subkey] = subval
    return merged

def extract_json_from_text(text):
    """
    Attempt to extract a JSON block from the text.
    First, look for content enclosed in triple backticks (optionally with "json"),
    then fall back to extracting from the first '{' to the last '}'.
    Returns the extracted JSON string if successful, or None if not found.
    """
    import re
    import json

    # Look for triple backticks encapsulating JSON.
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