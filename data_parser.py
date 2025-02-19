# data_parser.py
import json
import math
import logging
from ocr_utils import extract_json_from_text  # Make sure this is defined in ocr_utils.py

logger = logging.getLogger(__name__)
SCALE_FACTOR = 1  # Adjust as needed

def parse_value(value):
    """
    Convert a Brazilian-formatted number (as a string or number) to a float.
    If the value is a dict:
      - If it has exactly one key, use that value.
      - If it has multiple keys and the keys are numeric (i.e. years), choose the value corresponding to the highest year.
      - Otherwise, fall back to the first key's value.
    """
    if isinstance(value, dict):
        try:
            # Try to interpret keys as years (integers)
            numeric_keys = {}
            for k, v in value.items():
                try:
                    numeric_keys[int(k)] = v
                except Exception:
                    continue
            if numeric_keys:
                max_year = max(numeric_keys.keys())
                value = numeric_keys[max_year]
            else:
                # If not all keys are numeric, use the first value
                value = next(iter(value.values()))
        except Exception as e:
            logger.warning("Error processing dict value %s: %s. Using first value.", value, e)
            value = next(iter(value.values()))
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
    - Extract JSON text from the API response.
    - If the returned raw data is flat (its keys do not represent years), wrap it under "Ano Desconhecido."
    - Otherwise, if all keys are numeric (representing years), preserve that structure.
    - For each year and category, parse the value using parse_value.
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
        # Determine if keys represent years.
        if all(isinstance(k, str) and k.isdigit() for k in raw_data.keys()):
            # Already multi-year data, do nothing.
            pass
        else:
            # Otherwise, assume the data is flat and wrap it.
            raw_data = {"Ano Desconhecido": raw_data}
        
        formatted = {}
        for year, data in raw_data.items():
            formatted[year] = {}
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
    For each year, if a value is NaN in the merged data and a non-NaN value exists in a new result,
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