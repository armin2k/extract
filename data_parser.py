# data_parser.py
import math
import json
import re
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

def parse_value(value: Any) -> float:
    """
    Convert a Brazilian-formatted number (as string) to a float.
    """
    try:
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
            num = float(value)
            if negative:
                num = -num
            SCALE_FACTOR = 1  # Adjust if needed
            return num * SCALE_FACTOR
        return float(value)
    except Exception as e:
        logger.exception("Error parsing value %s: %s", value, e)
        return math.nan

def extract_json_from_text(text: str) -> str:
    """
    Extract a JSON block from text.
    """
    try:
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
            json.loads(candidate)
            return candidate
        return ""
    except Exception as e:
        logger.exception("Error extracting JSON: %s", e)
        return ""

def format_financial_data(response_json: Dict[str, Any], categories: list) -> Dict[str, Any]:
    """
    Convert the API response into a structured JSON object.
    """
    try:
        content = response_json.get('choices', [{}])[0].get('message', {}).get('content', '')
        if not content.strip():
            logger.error("API returned empty content.")
            return {}
        json_text = extract_json_from_text(content)
        if not json_text:
            logger.error("Could not extract JSON from API response.")
            return {}
        raw_data = json.loads(json_text)
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
        return {}

def merge_analysis_results(results: list, categories: list) -> Dict[str, Any]:
    """
    Merge multiple API responses into one JSON object.
    """
    merged = {}
    try:
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
    except Exception as e:
        logger.exception("Error merging analysis results: %s", e)
        return merged