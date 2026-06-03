from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.agent.schemas import Hypothesis
from src.config import LLMBackend
from src.utils.json_extract import extract_json_object


@dataclass
class ResultAnalysis:
    verdict: str
    confidence: float
    summary: str
    confounders: list[str]
    follow_ups: list[str]
    evidence_strength: float
    posterior: float
    raw: dict[str, Any]


def _numeric_digest(aggregated: dict[str, Any]) -> dict[str, Any]:
    digest: dict[str, Any] = {}
    for s in aggregated.get("steps", []):
        if not s.get("ok"):
            continue
        out = s.get("output", {})
        tool = s.get("tool")
        if tool == "differential_expression":
            digest["n_sig_q_0.05"] = out.get("n_significant_q_0.05")
        if tool == "pathway_enrichment":
            digest["n_enriched_pathways"] = len(out.get("enriched", []))
            digest["top_pathway"] = (
                out["enriched"][0] if out.get("enriched") else None
            )
        if tool == "profile_dataset":
            digest["profile"] = {k: out[k] for k in ("n_samples", "n_genes", "missing_fraction") if k in out}
    return digest


def analyze_results(
    llm: LLMBackend,
    hypothesis: Hypothesis,
    aggregated: dict[str, Any],
) -> ResultAnalysis:
    numbers = _numeric_digest(aggregated)
    prompt = f"""
Interpret computational results with scientific caution.

Hypothesis:
Statement: {hypothesis.statement}
Prediction: {hypothesis.testable_prediction}
Prior confidence: {hypothesis.confidence_prior}

Quantitative digest:
{json.dumps(numbers)[:3000]}

Step outputs (truncated): {json.dumps(aggregated.get("steps", []))[:3500]}

Return JSON:
{{
  "verdict": "supported" | "refuted" | "inconclusive",
  "confidence": 0.0-1.0,
  "summary": "...",
  "confounders": ["..."],
  "follow_ups": ["..."],
  "evidence_strength": 0.0-1.0
}}
"""
    text = llm.generate(prompt, system="Respond ONLY with valid JSON.", temperature=0.25)
    obj = extract_json_object(text)
    verdict = str(obj.get("verdict", "inconclusive"))
    conf = float(np.clip(float(obj.get("confidence", 0.5)), 0, 1))
    ev = float(np.clip(float(obj.get("evidence_strength", 0.5)), 0, 1))

    prior = float(np.clip(hypothesis.confidence_prior, 0.01, 0.99))
    lr_map = {"supported": 2.5, "refuted": 0.4, "inconclusive": 1.0}
    lr = lr_map.get(verdict, 1.0) * (0.4 + 0.6 * ev)
    odds = prior / (1 - prior)
    odds_post = odds * lr
    post = float(np.clip(odds_post / (1 + odds_post), 0.02, 0.98))

    return ResultAnalysis(
        verdict=verdict,
        confidence=conf,
        summary=str(obj.get("summary", "")),
        confounders=[str(x) for x in obj.get("confounders", [])],
        follow_ups=[str(x) for x in obj.get("follow_ups", [])],
        evidence_strength=ev,
        posterior=post,
        raw=obj,
    )
