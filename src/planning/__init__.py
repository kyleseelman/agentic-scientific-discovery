"""Long-horizon research strategy and meta-reasoning."""

from __future__ import annotations

__all__ = [
    "assess_progress",
    "ConsolidatedKnowledge",
    "ContinualLearner",
    "DriftEvent",
    "evaluate_hypothesis_portfolio",
    "MetaLearner",
    "StrategyAssessment",
    "StrategyRecommendation",
    "StrategyRecord",
    "ToolRecommendation",
]


def __getattr__(name: str):  # noqa: ANN202 – lazy re-exports to avoid circular imports
    if name in ("evaluate_hypothesis_portfolio",):
        from src.planning.evaluation import evaluate_hypothesis_portfolio
        return evaluate_hypothesis_portfolio
    if name in ("StrategyAssessment", "assess_progress"):
        import src.planning.strategy as _strat
        return getattr(_strat, name)
    if name in ("MetaLearner", "StrategyRecord", "StrategyRecommendation", "ToolRecommendation"):
        import src.planning.meta_learner as _ml
        return getattr(_ml, name)
    if name in ("ContinualLearner", "ConsolidatedKnowledge", "DriftEvent"):
        import src.planning.continual_learner as _cl
        return getattr(_cl, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
