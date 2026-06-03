from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Finding:
    id: str
    statement: str
    evidence_strength: float
    experiment_id: str
    hypothesis_id: str
    provenance: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now)
    entities: list[str] = field(default_factory=list)


@dataclass
class EntityRelation:
    id: str
    source: str
    relation: str
    target: str
    confidence: float
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class OpenQuestion:
    id: str
    question: str
    priority: float
    linked_hypotheses: list[str] = field(default_factory=list)


@dataclass
class HypothesisRecord:
    id: str
    statement: str
    rationale: str
    testable_prediction: str
    required_data: list[str]
    confidence_prior: float
    novelty_score: float
    status: str
    confidence_posterior: float | None = None
    updated_at: str = field(default_factory=_utc_now)


class KnowledgeStore:
    """JSON-backed knowledge base with provenance linking."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._hypotheses: dict[str, HypothesisRecord] = {}
        self._findings: dict[str, Finding] = {}
        self._relations: dict[str, EntityRelation] = {}
        self._questions: dict[str, OpenQuestion] = {}
        self._load()

    def _paths(self) -> dict[str, Path]:
        return {
            "hypotheses": self.root / "hypotheses.json",
            "findings": self.root / "findings.json",
            "relations": self.root / "relations.json",
            "questions": self.root / "open_questions.json",
        }

    def _load(self) -> None:
        p = self._paths()
        if p["hypotheses"].exists():
            raw = json.loads(p["hypotheses"].read_text())
            for k, v in raw.items():
                vd = dict(v)
                vd.setdefault("confidence_posterior", None)
                self._hypotheses[k] = HypothesisRecord(**vd)
        if p["findings"].exists():
            raw = json.loads(p["findings"].read_text())
            self._findings = {k: Finding(**v) for k, v in raw.items()}
        if p["relations"].exists():
            raw = json.loads(p["relations"].read_text())
            self._relations = {k: EntityRelation(**v) for k, v in raw.items()}
        if p["questions"].exists():
            raw = json.loads(p["questions"].read_text())
            self._questions = {k: OpenQuestion(**v) for k, v in raw.items()}

    def save(self) -> None:
        p = self._paths()
        p["hypotheses"].write_text(
            json.dumps({k: asdict(v) for k, v in self._hypotheses.items()}, indent=2)
        )
        p["findings"].write_text(
            json.dumps({k: asdict(v) for k, v in self._findings.items()}, indent=2)
        )
        p["relations"].write_text(
            json.dumps({k: asdict(v) for k, v in self._relations.items()}, indent=2)
        )
        p["questions"].write_text(
            json.dumps({k: asdict(v) for k, v in self._questions.items()}, indent=2)
        )

    def upsert_hypothesis(self, h: HypothesisRecord) -> None:
        h.updated_at = _utc_now()
        self._hypotheses[h.id] = h

    def update_hypothesis_status(
        self,
        hypothesis_id: str,
        status: str,
        posterior: float | None = None,
    ) -> None:
        if hypothesis_id not in self._hypotheses:
            return
        h = self._hypotheses[hypothesis_id]
        h.status = status
        h.updated_at = _utc_now()
        if posterior is not None:
            h.confidence_posterior = posterior

    def add_finding(self, finding: Finding) -> None:
        self._findings[finding.id] = finding

    def add_relation(self, rel: EntityRelation) -> None:
        self._relations[rel.id] = rel

    def add_open_question(self, q: OpenQuestion) -> None:
        self._questions[q.id] = q

    def get_hypothesis(self, hypothesis_id: str) -> HypothesisRecord | None:
        return self._hypotheses.get(hypothesis_id)

    def hypotheses_about_entity(self, entity: str) -> list[HypothesisRecord]:
        ent = entity.lower()
        out: list[HypothesisRecord] = []
        for h in self._hypotheses.values():
            if ent in h.statement.lower():
                out.append(h)
        return out

    def findings_about_entity(self, entity: str) -> list[Finding]:
        ent = entity.lower()
        return [f for f in self._findings.values() if ent in f.statement.lower()]

    def all_hypotheses(self) -> list[HypothesisRecord]:
        return list(self._hypotheses.values())

    def all_findings(self) -> list[Finding]:
        return list(self._findings.values())

    def summary_blob(self) -> str:
        lines: list[str] = []
        for h in list(self._hypotheses.values())[-8:]:
            post = (
                f"{h.confidence_posterior:.2f}"
                if h.confidence_posterior is not None
                else "n/a"
            )
            lines.append(
                f"- [{h.status}] {h.statement} (prior={h.confidence_prior:.2f}, post={post})"
            )
        for f in list(self._findings.values())[-8:]:
            lines.append(
                f"- finding[{f.evidence_strength:.2f}]: {f.statement} "
                f"(exp={f.experiment_id}, hyp={f.hypothesis_id})"
            )
        return "\n".join(lines) if lines else "(empty knowledge store)"


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"
