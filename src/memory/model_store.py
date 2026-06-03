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


class ModelStore:
    """Track trained models, their metrics, and lineage across research cycles."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records: list[ModelRecord] = []
        if self.path.exists():
            raw = json.loads(self.path.read_text())
            self._records = [ModelRecord(**r) for r in raw.get("records", [])]

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
