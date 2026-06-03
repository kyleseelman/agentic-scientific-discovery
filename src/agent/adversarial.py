"""Adversarial hypothesis testing.

Provides a structured falsification protocol that runs after the initial
analysis to stress-test conclusions before they are accepted. Uses the
configured LLM backend with an adversarial system prompt.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.agent.schemas import Hypothesis, ExperimentPlan, ExperimentStep
from src.agent.result_analyzer import ResultAnalysis
from src.config import LLMBackend
from src.utils.json_extract import extract_json_object


@dataclass
class FalsificationExperiment:
    """An experiment specifically designed to disprove a hypothesis."""
    description: str
    null_hypothesis: str
    tool_steps: list[ExperimentStep]
    expected_if_false: str


@dataclass
class AdversarialReview:
    """Result of adversarial review of a hypothesis + evidence."""
    objections: list[str]
    severity: str  # "minor", "moderate", "serious", "fatal"
    falsification_experiments: list[FalsificationExperiment]
    alternative_explanations: list[str]
    statistical_concerns: list[str]
    confidence_adjustment: float  # multiply posterior by this (< 1.0 = less confident)
    recommendation: str  # "accept", "revise", "retest", "reject"
    reasoning: str


_ADVERSARIAL_SYSTEM = """You are a rigorous scientific reviewer whose job is to find flaws \
in hypotheses and experimental evidence. You are skeptical by default. \
You check for: confounders, insufficient sample sizes, multiple testing \
issues, overfitting risk, batch effects, selection bias, p-hacking, and \
conclusions that go beyond what the data supports. Always suggest at \
least one falsification experiment. Respond ONLY with valid JSON."""


def adversarial_review(
    llm: LLMBackend,
    hypothesis: Hypothesis,
    analysis: ResultAnalysis,
    experiment_trace: list[dict[str, Any]],
) -> AdversarialReview:
    """Run adversarial review on a hypothesis after initial analysis.

    The LLM adopts a devil's-advocate persona and tries to find flaws
    in the reasoning, statistical methodology, and conclusions.
    """
    prompt = f"""
Critically review this hypothesis and the evidence for it. Try to DISPROVE it.

Hypothesis:
  Statement: {hypothesis.statement}
  Prediction: {hypothesis.testable_prediction}
  Prior: {hypothesis.confidence_prior}

Analysis verdict: {analysis.verdict}
Analysis confidence: {analysis.confidence}
Evidence strength: {analysis.evidence_strength}
Summary: {analysis.summary}
Confounders already noted: {json.dumps(analysis.confounders)}

Experiment trace (tools used and results):
{json.dumps(experiment_trace)[:4000]}

Return JSON:
{{
  "objections": ["list of specific scientific objections"],
  "severity": "minor" | "moderate" | "serious" | "fatal",
  "alternative_explanations": ["plausible alternative explanations for the results"],
  "statistical_concerns": ["specific statistical methodology issues"],
  "falsification_experiments": [
    {{
      "description": "what to do",
      "null_hypothesis": "what to test against",
      "tools": ["tool1", "tool2"],
      "expected_if_false": "what result would disprove the hypothesis"
    }}
  ],
  "confidence_adjustment": 0.0-1.0,
  "recommendation": "accept" | "revise" | "retest" | "reject",
  "reasoning": "overall assessment"
}}

Rules:
- Be skeptical. Default to "revise" or "retest" unless evidence is very strong.
- Always find at least 2 objections.
- Always suggest at least 1 falsification experiment.
- confidence_adjustment < 0.8 means you found real problems.
- If verdict was "supported" with weak evidence, recommend "retest".
"""
    text = llm.generate(prompt, system=_ADVERSARIAL_SYSTEM, temperature=0.3)
    obj = extract_json_object(text)

    falsification_exps: list[FalsificationExperiment] = []
    for fe in obj.get("falsification_experiments", []):
        steps = [
            ExperimentStep(tool=t, params={}, description="")
            for t in fe.get("tools", [])
        ]
        falsification_exps.append(FalsificationExperiment(
            description=str(fe.get("description", "")),
            null_hypothesis=str(fe.get("null_hypothesis", "")),
            tool_steps=steps,
            expected_if_false=str(fe.get("expected_if_false", "")),
        ))

    if not falsification_exps:
        falsification_exps.append(FalsificationExperiment(
            description="Permutation test on group labels",
            null_hypothesis="Group assignment has no effect on the observed pattern",
            tool_steps=[ExperimentStep("cross_validate_hypothesis", {}, "Permutation test")],
            expected_if_false="Permutation p-value > 0.05",
        ))

    adj = float(np.clip(float(obj.get("confidence_adjustment", 0.8)), 0.1, 1.0))

    return AdversarialReview(
        objections=[str(o) for o in obj.get("objections", ["No specific objections generated"])],
        severity=str(obj.get("severity", "moderate")),
        falsification_experiments=falsification_exps,
        alternative_explanations=[str(a) for a in obj.get("alternative_explanations", [])],
        statistical_concerns=[str(s) for s in obj.get("statistical_concerns", [])],
        confidence_adjustment=adj,
        recommendation=str(obj.get("recommendation", "retest")),
        reasoning=str(obj.get("reasoning", "")),
    )


def apply_adversarial_adjustment(
    analysis: ResultAnalysis,
    review: AdversarialReview,
) -> ResultAnalysis:
    """Create an updated ResultAnalysis with adversarial adjustments applied."""
    adjusted_posterior = float(np.clip(
        analysis.posterior * review.confidence_adjustment, 0.02, 0.98
    ))

    all_confounders = list(set(analysis.confounders + review.alternative_explanations))

    follow_ups = list(analysis.follow_ups)
    for fe in review.falsification_experiments:
        follow_ups.append(f"[FALSIFICATION] {fe.description}: {fe.null_hypothesis}")

    verdict = analysis.verdict
    if review.recommendation == "reject" and analysis.verdict == "supported":
        verdict = "inconclusive"
    elif review.severity == "fatal":
        verdict = "inconclusive"

    raw = dict(analysis.raw)
    raw["adversarial_review"] = {
        "objections": review.objections,
        "severity": review.severity,
        "confidence_adjustment": review.confidence_adjustment,
        "recommendation": review.recommendation,
        "reasoning": review.reasoning,
        "n_falsification_experiments": len(review.falsification_experiments),
    }

    return ResultAnalysis(
        verdict=verdict,
        confidence=analysis.confidence * review.confidence_adjustment,
        summary=analysis.summary,
        confounders=all_confounders,
        follow_ups=follow_ups,
        evidence_strength=analysis.evidence_strength,
        posterior=adjusted_posterior,
        raw=raw,
    )
