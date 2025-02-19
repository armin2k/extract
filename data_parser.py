# data_parser.py
import json
import math
import logging
from ocr_utils import extract_json_from_text  # Assuming you have a helper to extract JSON from text

logger = logging.getLogger(__name__)

# You may have a SCALE_FACTOR defined elsewhere; set it here if not.
SCALE_FACTOR = 1

def parse_value(value):
    """
    Convert a Brazilian-formatted number string to a float.
    If the value is a dict (unexpected), try to extract a numeric value from it.
    """
    # If value is a dict, attempt to extract its numeric content.
    if isinstance(value, dict):
        # Option 1: If the dict has a 'value' key, use that.
        if 'value' in value:
            value = value['value']
        # Option 2: If the dict has exactly one key, use its value.
        elif len(value) == 1:
            value = next(iter(value.values()))
        else:
            # If the dict has multiple keys, log a warning and return NaN.
            logger.warning("Multiple keys in value dict: %s. Returning NaN.", value)
            return math.nan
    # Now attempt to convert to float.
    if value in [None, "", "NaN"]:
        return math.nan
    if isinstance(value, str):
        try:
            # Remove "R$", spaces, etc. Adjust for Brazilian number formats if needed.
            value = value.replace("R$", "").strip()
            # Remove any thousand separators and replace decimal comma with dot.
            value = value.replace(".", "").replace(",", ".")
            num = float(value)
            return num * SCALE_FACTOR
        except Exception as e:
            logger.exception("Error converting value %s: %s", value, e)
            return math.nan
    try:
        return float(value) * SCALE_FACTOR
    except Exception as e:
        logger.exception("Error converting value %s: %s", value, e)
        return math.nan

def format_financial_data(response_json, categories):
    """
    Convert the API response into a structured JSON object.
    Expects the API to return a JSON block containing financial data.
    If the returned data is flat and matches categories, it wraps it under "Ano Desconhecido".
    Then, it parses each value using parse_value.
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
        # If raw_data is flat and keys match categories, wrap it under "Ano Desconhecido"
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
        logger.exception("Error formatting financial data: %s", e)
        return None

def merge_analysis_results(results, categories):
    """
    Merge multiple analysis result dictionaries.
    For each year, prefer non-NaN values from the new results.
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
    This can be implemented by looking for content within triple backticks or from the first '{' to the last '}'.
    """
    import re
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