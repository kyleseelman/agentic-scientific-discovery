from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.config import AppConfig, LLMBackend, get_config
from src.utils.json_extract import extract_json_object


@dataclass
class StrategyAssessment:
    continue_thread: bool
    reason: str
    suggested_focus: str
    raw: dict[str, Any]


def assess_progress(
    llm: LLMBackend,
    research_question: str,
    knowledge_summary: str,
    cycles_completed: int,
    config: AppConfig | None = None,
) -> StrategyAssessment:
    cfg = config or get_config()
    prompt = f"""
You coordinate autonomous scientific discovery.
Research question: {research_question}
Cycles completed: {cycles_completed}

Current knowledge summary:
{knowledge_summary}

Respond with JSON only:
{{
  "continue_thread": true/false,
  "reason": "...",
  "suggested_focus": "..."
}}
"""
    text = llm.generate(prompt, system="Return strictly valid JSON.", temperature=0.3)
    obj = extract_json_object(text)
    return StrategyAssessment(
        continue_thread=bool(obj.get("continue_thread", True)),
        reason=str(obj.get("reason", "")),
        suggested_focus=str(obj.get("suggested_focus", "")),
        raw=obj,
    )
