"""Track trained ML models, their metrics, and lineage across research cycles."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ModelRecord:
    id: str
    model_type: str
    task: str
    hypothesis_id: str | None
    experiment_id: str | None
    metrics: dict[str, float]
    hyperparameters: dict
    feature_genes: list[str]
    model_path: str
    training_time_s: float
    created_at: str = field(default_factory=_utc_now)
    notes: str = ""
    baseline: bool = False
    paper_reference: str = ""
    comparison_to_baselines: dict[str, float] = field(default_factory=dict)


class ModelStore:
    """Track trained models, their metrics, and lineage across research cycles."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records: list[ModelRecord] = []
        if self.path.exists():
            raw = json.loads(self.path.read_text())
            for r in raw.get("records", []):
                r.setdefault("baseline", False)
                r.setdefault("paper_reference", "")
                r.setdefault("comparison_to_baselines", {})
                self._records.append(ModelRecord(**r))

    def add(self, record: ModelRecord) -> None:
        self._records.append(record)
        self._flush()

    def _flush(self) -> None:
        payload = {"records": [asdict(r) for r in self._records]}
        self.path.write_text(json.dumps(payload, indent=2))

    def all_models(self) -> list[ModelRecord]:
        return list(self._records)

    def best_model(self, task: str, metric: str = "accuracy") -> ModelRecord | None:
        candidates = [r for r in self._records if r.task == task and metric in r.metrics]
        if not candidates:
            return None
        return max(candidates, key=lambda r: r.metrics[metric])

    def models_for_hypothesis(self, hypothesis_id: str) -> list[ModelRecord]:
        return [r for r in self._records if r.hypothesis_id == hypothesis_id]

    def recent(self, n: int = 10) -> list[ModelRecord]:
        return self._records[-n:]

    def all_for_task(self, task: str) -> list[ModelRecord]:
        """Retrieve all models (including baselines) for a given task string."""
        return [r for r in self._records if task.lower() in r.task.lower()]

    def sota_summary(self, task: str, metric: str = "accuracy") -> dict:
        """Return best model + all baselines sorted by metric for a task."""
        task_models = self.all_for_task(task)
        if not task_models:
            return {"best": None, "baselines": [], "all_ranked": []}

        with_metric = [r for r in task_models if metric in r.metrics]
        if not with_metric:
            return {"best": None, "baselines": [asdict(r) for r in task_models], "all_ranked": []}

        ranked = sorted(with_metric, key=lambda r: r.metrics[metric], reverse=True)
        baselines = [r for r in ranked if r.baseline]
        novel = [r for r in ranked if not r.baseline]

        return {
            "best": asdict(ranked[0]),
            "best_is_baseline": ranked[0].baseline,
            "baselines": [asdict(r) for r in baselines],
            "novel_models": [asdict(r) for r in novel],
            "all_ranked": [
                {"id": r.id, "type": r.model_type, "baseline": r.baseline,
                 metric: r.metrics[metric]}
                for r in ranked
            ],
        }
