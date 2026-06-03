from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ExperimentRecord:
    id: str
    hypothesis_id: str
    plan: dict[str, Any]
    execution_trace: list[dict[str, Any]]
    results: dict[str, Any]
    interpretation: dict[str, Any]
    created_at: str = field(default_factory=_utc_now)


class ExperimentLog:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._records: list[ExperimentRecord] = []
        if self.path.exists():
            raw = json.loads(self.path.read_text())
            self._records = [ExperimentRecord(**r) for r in raw.get("records", [])]

    def append(self, record: ExperimentRecord) -> None:
        self._records.append(record)
        self._flush()

    def _flush(self) -> None:
        payload = {"records": [asdict(r) for r in self._records]}
        self.path.write_text(json.dumps(payload, indent=2))

    def recent(self, n: int = 10) -> list[ExperimentRecord]:
        return self._records[-n:]

    def all(self) -> list[ExperimentRecord]:
        return list(self._records)
