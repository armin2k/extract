# data_parser.py
import json
import math
import logging
from ocr_utils import extract_json_from_text  # Ensure this is defined in ocr_utils.py

logger = logging.getLogger(__name__)
SCALE_FACTOR = 1  # Adjust if necessary

def parse_value(value):
    """
    Convert a Brazilian-formatted number (as a string or number) to a float.
    If the value is a dict, this function should not be called directly;
    instead, the caller should iterate over its values.
    """
    # If the value is already a number (int/float), return it.
    if isinstance(value, (int, float)):
        return float(value) * SCALE_FACTOR
    if value in [None, "", "NaN"]:
        return math.nan
    try:
        if isinstance(value, str):
            # Remove currency symbols and spaces.
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
    
    - Extracts the JSON block from the API response.
    - If the raw data is flat (keys do not represent years), wraps it under "Ano Desconhecido".
    - If the raw data already has year keys, that structure is preserved.
    - For each year and each category, if the value is a dictionary (i.e. multi-year sub-values),
      each inner value is converted using parse_value. Otherwise, the value is converted directly.
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
        # Determine if the raw data keys represent years.
        if isinstance(raw_data, dict) and all(isinstance(k, str) and k.strip().isdigit() for k in raw_data.keys()):
            # Data is multi-year; preserve structure.
            multi_year = True
        else:
            multi_year = False
            raw_data = {"Ano Desconhecido": raw_data}
        
        formatted = {}
        for year, data in raw_data.items():
            formatted[year] = {}
            for category in categories:
                if isinstance(data, dict):
                    raw_value = data.get(category, math.nan)
                else:
                    raw_value = data
                # If the raw_value itself is a dictionary, process each inner value.
                if isinstance(raw_value, dict):
                    formatted_value = {subkey: parse_value(subval) for subkey, subval in raw_value.items()}
                else:
                    formatted_value = parse_value(raw_value)
                formatted[year][category] = formatted_value
        return formatted
    except Exception as e:
        logger.exception("Error formatting financial data: %s", e)
        return None

def merge_analysis_results(results, categories):
    """
    Merge multiple analysis result dictionaries.
    For each year and category, if the current merged value is NaN and a new result provides a non-NaN value,
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
                    # If current value is a simple number and new_val is a number, use new_val if current is NaN.
                    if not isinstance(current_val, dict) and math.isnan(current_val) and not math.isnan(new_val):
                        merged[year][category] = new_val
                    # If both are dicts, merge them (here simply prefer non-NaN new values)
                    elif isinstance(current_val, dict) and isinstance(new_val, dict):
                        for subkey, subval in new_val.items():
                            if math.isnan(current_val.get(subkey, math.nan)) and not math.isnan(subval):
                                merged[year][category][subkey] = subval
    return merged

def extract_json_from_text(text):
    """
    Attempt to extract a JSON block from the text.
    
    This function looks first for content enclosed in triple backticks (optionally with "json"),
    then falls back to extracting from the first '{' to the last '}'.
    Returns the extracted JSON string if successful, or None if no valid JSON is found.
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

    # Fallback: Extract from first '{' to last '}'
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