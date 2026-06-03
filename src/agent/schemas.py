from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Hypothesis:
    id: str
    statement: str
    rationale: str
    testable_prediction: str
    required_data: list[str]
    confidence_prior: float
    novelty_score: float
    status: str
    literature_grounded: bool = False
    source_papers: list[str] = field(default_factory=list)


@dataclass
class ExperimentStep:
    tool: str
    params: dict[str, Any]
    description: str


@dataclass
class ExperimentPlan:
    hypothesis_id: str
    steps: list[ExperimentStep]
    expected_duration: str
    success_criteria: str
    failure_criteria: str
