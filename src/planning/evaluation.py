from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from src.agent.schemas import Hypothesis


@dataclass
class HypothesisScore:
    hypothesis_id: str
    information_gain: float
    feasibility: float
    novelty: float
    impact: float
    combined: float


def _uncertainty_gain(prior: float) -> float:
    p = float(np.clip(prior, 1e-3, 1 - 1e-3))
    # Binary entropy as proxy for epistemic uncertainty
    return float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))


def score_hypothesis(h: Hypothesis, available_data_keys: set[str]) -> HypothesisScore:
    req = set(h.required_data)
    overlap = len(req & available_data_keys)
    feasibility = overlap / max(1, len(req))
    ig = _uncertainty_gain(h.confidence_prior)
    novelty = float(np.clip(h.novelty_score, 0, 1))
    impact = float(np.clip(h.confidence_prior * 0.5 + novelty * 0.5, 0, 1))
    combined = 0.45 * ig + 0.35 * feasibility + 0.15 * novelty + 0.05 * impact
    return HypothesisScore(
        hypothesis_id=h.id,
        information_gain=ig,
        feasibility=feasibility,
        novelty=novelty,
        impact=impact,
        combined=combined,
    )


def evaluate_hypothesis_portfolio(
    hypotheses: Iterable[Hypothesis],
    available_data_keys: set[str],
) -> list[HypothesisScore]:
    return sorted(
        [score_hypothesis(h, available_data_keys) for h in hypotheses],
        key=lambda s: s.combined,
        reverse=True,
    )
