# data_parser.py
import json
import math
import logging
from ocr_utils import extract_json_from_text  # Ensure this exists in ocr_utils.py

logger = logging.getLogger(__name__)
SCALE_FACTOR = 1  # Adjust as necessary

def parse_value(value):
    """
    Convert a Brazilian-formatted number (as a string or number) to a float.
    If the value is a dict:
      - If it has exactly one key, use its value.
      - Otherwise, log a warning and return NaN.
    """
    if isinstance(value, dict):
        if len(value) == 1:
            value = next(iter(value.values()))
        else:
            logger.warning("Multiple keys in value dict: %s. Returning NaN.", value)
            return math.nan
    if value in [None, "", "NaN"]:
        return math.nan
    try:
        if isinstance(value, str):
            # Remove any currency symbols and spaces.
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
    - Extracts JSON text from the response content.
    - If raw_data is flat (keys matching categories), wraps it under "Ano Desconhecido".
    - Then for each year and category, parses the value with parse_value.
    Handles cases where the value might be a simple number or a dict.
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
        # Wrap flat data in a default key if needed.
        if any(key in categories for key in raw_data.keys()):
            raw_data = {"Ano Desconhecido": raw_data}
        formatted = {}
        for year, data in raw_data.items():
            formatted[year] = {}
            for category in categories:
                # If data is not a dict, assume it's a simple value.
                if isinstance(data, dict):
                    raw_value = data.get(category, math.nan)
                else:
                    raw_value = data
                if not isinstance(raw_value, dict):
                    formatted[year][category] = parse_value(raw_value)
                else:
                    if len(raw_value) == 1:
                        formatted[year][category] = parse_value(next(iter(raw_value.values())))
                    else:
                        logger.warning("Multiple keys in value dict for category '%s': %s. Returning NaN.", category, raw_value)
                        formatted[year][category] = math.nan
        return formatted
    except Exception as e:
        logger.exception("Error formatting financial data: %s", e)
        return None

def merge_analysis_results(results, categories):
    """
    Merge multiple analysis result dictionaries.
    For each accounting year, use non-NaN values from later results if available.
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

# The following helper function attempts to extract a JSON block from a text string.
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