import json
import math
import logging
from ocr_utils import extract_json_from_text

logger = logging.getLogger(__name__)
SCALE_FACTOR = 1

def parse_value(value) -> float:
    """
    Convert a Brazilian-formatted number (as a string) to a float.
    If the value is a dict with exactly one key, use that value;
    otherwise, log a warning and return NaN.
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
            value = value.replace("R$", "").strip()
            value = value.replace(".", "").replace(",", ".")
        return float(value) * SCALE_FACTOR
    except Exception as e:
        logger.exception("Error converting value %s: %s", value, e)
        return math.nan

def extract_json_from_text(text: str) -> str:
    import re
    import json

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
            logger.error("Failed to decode JSON from extracted candidate.")
            return None
    return None

def format_financial_data(response_json: dict, categories: list) -> dict:
    """
    Convert the API response into a structured JSON object.
    If the API returns a flat dictionary (its keys match categories), wrap it under "Ano Desconhecido".
    """
    try:
        content = response_json.get('choices', [{}])[0].get('message', {}).get('content', '')
        if not content.strip():
            logger.error("Empty content in API response.")
            return None
        json_text = extract_json_from_text(content)
        if not json_text:
            logger.error("Could not extract JSON from API response.")
            logger.error("Raw response snippet: %s", content[:500])
            return None
        try:
            raw_data = json.loads(json_text)
        except json.JSONDecodeError as jde:
            logger.error("Failed to decode JSON. Error: %s", jde)
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
        logger.error(f"Error formatting financial data: {e}")
        return None

def merge_analysis_results(results, categories) -> dict:
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