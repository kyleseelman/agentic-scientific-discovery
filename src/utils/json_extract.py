from __future__ import annotations

import json
import re
from typing import Any


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse first JSON object from model output; tolerates markdown fences."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("No JSON object found in model output")


def extract_json_array(text: str) -> list[Any]:
    obj = extract_json_object(text)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and "hypotheses" in obj:
        return list(obj["hypotheses"])
    raise ValueError("Expected JSON array or object with 'hypotheses' key")
