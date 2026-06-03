from __future__ import annotations

import json
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import LLMBackend
from src.memory.knowledge_store import Finding, KnowledgeStore


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ConsolidatedKnowledge:
    topic: str
    findings: list[str]
    confidence: float
    supporting_sessions: list[str]
    contradicting_sessions: list[str]
    last_updated: str = field(default_factory=_utc_now)


@dataclass
class DriftEvent:
    id: str
    topic: str
    old_belief: str
    new_evidence: str
    drift_type: str  # "contradiction", "refinement", "extension"
    detected_at: str = field(default_factory=_utc_now)
    resolved: bool = False
    resolution: str = ""


class ContinualLearner:
    """Cross-session adaptation and knowledge consolidation."""

    def __init__(
        self,
        store_path: Path | str,
        llm: LLMBackend | None = None,
    ) -> None:
        self._store_path = Path(store_path)
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        self._llm = llm
        self._consolidated: list[ConsolidatedKnowledge] = []
        self._drift_events: list[DriftEvent] = []
        self._priors: dict[str, dict[str, float]] = {}
        self._load()

    def _load(self) -> None:
        if self._store_path.exists():
            raw = json.loads(self._store_path.read_text())
            self._consolidated = [
                ConsolidatedKnowledge(**c) for c in raw.get("consolidated", [])
            ]
            self._drift_events = [
                DriftEvent(**d) for d in raw.get("drift_events", [])
            ]
            self._priors = raw.get("priors", {})

    def _save(self) -> None:
        payload = {
            "consolidated": [asdict(c) for c in self._consolidated],
            "drift_events": [asdict(d) for d in self._drift_events],
            "priors": self._priors,
        }
        self._store_path.write_text(json.dumps(payload, indent=2))

    def consolidate_session(
        self,
        knowledge_store: KnowledgeStore,
        session_id: str,
    ) -> None:
        """Merge a session's findings into consolidated knowledge.

        Groups findings by topic (embedding similarity when available,
        keyword overlap otherwise), detects contradictions, and updates
        confidence scores based on accumulated evidence.
        """
        findings = knowledge_store.all_findings()
        if not findings:
            return

        topic_groups = self._group_by_topic(findings)

        for topic, group in topic_groups.items():
            existing = self._find_consolidated(topic)
            new_stmts = [f.statement for f in group]

            if existing is not None:
                for stmt in new_stmts:
                    drift_events = self._check_contradiction(existing, stmt)
                    self._drift_events.extend(drift_events)
                    if stmt not in existing.findings:
                        existing.findings.append(stmt)

                if session_id not in existing.supporting_sessions:
                    if any(d.drift_type == "contradiction" for d in drift_events):
                        if session_id not in existing.contradicting_sessions:
                            existing.contradicting_sessions.append(session_id)
                    else:
                        existing.supporting_sessions.append(session_id)

                avg_strength = sum(f.evidence_strength for f in group) / len(group)
                n_support = len(existing.supporting_sessions)
                n_contra = len(existing.contradicting_sessions)
                existing.confidence = round(
                    avg_strength * n_support / max(n_support + n_contra, 1), 4
                )
                existing.last_updated = _utc_now()
            else:
                avg_strength = sum(f.evidence_strength for f in group) / len(group)
                self._consolidated.append(
                    ConsolidatedKnowledge(
                        topic=topic,
                        findings=new_stmts,
                        confidence=round(avg_strength, 4),
                        supporting_sessions=[session_id],
                        contradicting_sessions=[],
                    )
                )

        self._save()

    def _group_by_topic(
        self,
        findings: list[Finding],
    ) -> dict[str, list[Finding]]:
        """Group findings by topic using embedding similarity or keyword overlap."""
        embedder = self._get_embedder()
        if embedder is not None:
            return self._group_by_embedding(findings, embedder)
        return self._group_by_keywords(findings)

    def _get_embedder(self) -> Any:
        """Try to load a sentence-transformer for embedding-based grouping."""
        try:
            from sentence_transformers import SentenceTransformer
            return SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            return None

    def _group_by_embedding(
        self,
        findings: list[Finding],
        embedder: Any,
    ) -> dict[str, list[Finding]]:
        """Cluster findings by cosine similarity of their embeddings."""
        import numpy as np

        statements = [f.statement for f in findings]
        embeddings = embedder.encode(statements, convert_to_numpy=True)

        threshold = 0.65
        assigned = [False] * len(findings)
        groups: dict[str, list[Finding]] = {}

        for i in range(len(findings)):
            if assigned[i]:
                continue
            cluster = [findings[i]]
            assigned[i] = True
            for j in range(i + 1, len(findings)):
                if assigned[j]:
                    continue
                sim = float(
                    np.dot(embeddings[i], embeddings[j])
                    / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-9)
                )
                if sim >= threshold:
                    cluster.append(findings[j])
                    assigned[j] = True
            topic = self._extract_topic(cluster[0].statement)
            groups[topic] = cluster

        return groups

    def _group_by_keywords(
        self,
        findings: list[Finding],
    ) -> dict[str, list[Finding]]:
        """Fallback grouping via Jaccard similarity on word tokens."""
        _STOP = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "of", "in", "to", "and", "or", "for", "with", "on", "by",
            "that", "this", "it", "its", "from", "as", "at", "not", "no",
        }
        threshold = 0.25

        def tokens(text: str) -> set[str]:
            return {w.lower().strip(".,;:!?") for w in text.split()} - _STOP

        assigned = [False] * len(findings)
        groups: dict[str, list[Finding]] = {}

        for i in range(len(findings)):
            if assigned[i]:
                continue
            cluster = [findings[i]]
            assigned[i] = True
            ti = tokens(findings[i].statement)
            for j in range(i + 1, len(findings)):
                if assigned[j]:
                    continue
                tj = tokens(findings[j].statement)
                union = ti | tj
                if not union:
                    continue
                if len(ti & tj) / len(union) >= threshold:
                    cluster.append(findings[j])
                    assigned[j] = True
            topic = self._extract_topic(cluster[0].statement)
            groups[topic] = cluster

        return groups

    @staticmethod
    def _extract_topic(statement: str) -> str:
        """Derive a short topic label from the first finding statement."""
        words = statement.split()
        return " ".join(words[:8]).rstrip(".,;:")

    def _find_consolidated(self, topic: str) -> ConsolidatedKnowledge | None:
        """Find existing consolidated knowledge matching a topic."""
        topic_lower = topic.lower()
        for ck in self._consolidated:
            existing_lower = ck.topic.lower()
            overlap = set(topic_lower.split()) & set(existing_lower.split())
            union = set(topic_lower.split()) | set(existing_lower.split())
            if union and len(overlap) / len(union) > 0.3:
                return ck
        return None

    def _check_contradiction(
        self,
        existing: ConsolidatedKnowledge,
        new_statement: str,
    ) -> list[DriftEvent]:
        """Detect contradictions between existing knowledge and a new finding."""
        events: list[DriftEvent] = []
        _NEGATION_PAIRS = [
            ("increase", "decrease"),
            ("upregulated", "downregulated"),
            ("up-regulated", "down-regulated"),
            ("higher", "lower"),
            ("positive", "negative"),
            ("activates", "inhibits"),
            ("promotes", "suppresses"),
            ("supported", "refuted"),
        ]
        new_lower = new_statement.lower()
        for old_finding in existing.findings:
            old_lower = old_finding.lower()
            old_tokens = set(old_lower.split())
            new_tokens = set(new_lower.split())
            if len(old_tokens & new_tokens) / max(len(old_tokens | new_tokens), 1) < 0.2:
                continue

            for pos, neg in _NEGATION_PAIRS:
                if (pos in old_lower and neg in new_lower) or (
                    neg in old_lower and pos in new_lower
                ):
                    events.append(
                        DriftEvent(
                            id=f"drift_{uuid.uuid4().hex[:10]}",
                            topic=existing.topic,
                            old_belief=old_finding,
                            new_evidence=new_statement,
                            drift_type="contradiction",
                        )
                    )
                    break
            else:
                if len(new_tokens - old_tokens) > len(new_tokens) * 0.5:
                    events.append(
                        DriftEvent(
                            id=f"drift_{uuid.uuid4().hex[:10]}",
                            topic=existing.topic,
                            old_belief=old_finding,
                            new_evidence=new_statement,
                            drift_type="extension",
                        )
                    )

        return events

    def detect_drift(self, new_finding: Finding) -> list[DriftEvent]:
        """Check if a new finding contradicts or refines existing knowledge."""
        events: list[DriftEvent] = []
        for ck in self._consolidated:
            events.extend(self._check_contradiction(ck, new_finding.statement))
        if events:
            self._drift_events.extend(events)
            self._save()
        return events

    def get_priors_for_topic(self, topic: str) -> dict[str, Any]:
        """Return accumulated priors for a research topic.

        Used by hypothesis generator to set better initial confidences.
        """
        ck = self._find_consolidated(topic)
        if ck is None:
            return {
                "confidence": 0.5,
                "n_findings": 0,
                "n_supporting_sessions": 0,
                "n_contradicting_sessions": 0,
            }
        return {
            "confidence": ck.confidence,
            "n_findings": len(ck.findings),
            "n_supporting_sessions": len(ck.supporting_sessions),
            "n_contradicting_sessions": len(ck.contradicting_sessions),
            "recent_findings": ck.findings[-5:],
        }

    def resolve_contradiction(
        self,
        drift_event_id: str,
        resolution: str,
    ) -> None:
        """Mark a contradiction as resolved with explanation."""
        for event in self._drift_events:
            if event.id == drift_event_id:
                event.resolved = True
                event.resolution = resolution
                break
        self._save()

    def knowledge_summary(self) -> str:
        """Summary of all consolidated knowledge for the LLM context window."""
        if not self._consolidated:
            return "No consolidated knowledge yet."

        lines: list[str] = [
            f"Consolidated knowledge ({len(self._consolidated)} topics):"
        ]
        for ck in self._consolidated:
            n_findings = len(ck.findings)
            n_sup = len(ck.supporting_sessions)
            n_con = len(ck.contradicting_sessions)
            lines.append(
                f"  [{ck.topic}] conf={ck.confidence:.2f}, "
                f"{n_findings} findings, {n_sup} supporting / {n_con} contradicting sessions"
            )
            for stmt in ck.findings[-3:]:
                lines.append(f"    - {stmt[:120]}")

        unresolved = [d for d in self._drift_events if not d.resolved]
        if unresolved:
            lines.append(f"\n  Unresolved drift events ({len(unresolved)}):")
            for d in unresolved[-5:]:
                lines.append(
                    f"    [{d.drift_type}] {d.topic}: "
                    f"old='{d.old_belief[:80]}' vs new='{d.new_evidence[:80]}'"
                )

        return "\n".join(lines)

    def update_priors(
        self,
        hypothesis_type: str,
        outcomes: list[str],
    ) -> float:
        """Bayesian update of prior confidence for a hypothesis type.

        Uses a Beta distribution: each "supported" outcome increments alpha,
        each "refuted" outcome increments beta. Returns the posterior mean.
        """
        from scipy.stats import beta  # type: ignore[import-untyped]

        prior = self._priors.get(hypothesis_type, {"alpha": 1.0, "beta": 1.0})
        alpha = prior["alpha"]
        beta_param = prior["beta"]

        for outcome in outcomes:
            if outcome == "supported":
                alpha += 1.0
            elif outcome == "refuted":
                beta_param += 1.0
            else:
                alpha += 0.3
                beta_param += 0.3

        self._priors[hypothesis_type] = {"alpha": alpha, "beta": beta_param}
        self._save()

        return float(beta.mean(alpha, beta_param))

    def cross_session_report(self) -> dict[str, Any]:
        """Compare findings across sessions, highlighting agreements and conflicts."""
        session_topics: dict[str, list[str]] = defaultdict(list)
        for ck in self._consolidated:
            for sid in ck.supporting_sessions:
                session_topics[sid].append(ck.topic)
            for sid in ck.contradicting_sessions:
                session_topics[sid].append(ck.topic)

        agreements: list[dict[str, Any]] = []
        conflicts: list[dict[str, Any]] = []

        for ck in self._consolidated:
            if len(ck.supporting_sessions) >= 2:
                agreements.append({
                    "topic": ck.topic,
                    "confidence": ck.confidence,
                    "sessions": ck.supporting_sessions,
                    "n_findings": len(ck.findings),
                })
            if ck.contradicting_sessions:
                conflicts.append({
                    "topic": ck.topic,
                    "supporting": ck.supporting_sessions,
                    "contradicting": ck.contradicting_sessions,
                })

        unresolved = [
            asdict(d) for d in self._drift_events if not d.resolved
        ]
        resolved = [
            asdict(d) for d in self._drift_events if d.resolved
        ]

        return {
            "total_topics": len(self._consolidated),
            "total_sessions": len(session_topics),
            "agreements": agreements,
            "conflicts": conflicts,
            "unresolved_drift": unresolved,
            "resolved_drift": resolved,
            "priors": dict(self._priors),
        }
