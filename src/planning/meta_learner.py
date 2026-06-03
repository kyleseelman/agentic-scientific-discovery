from __future__ import annotations

import json
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class StrategyRecord:
    strategy_id: str
    research_question: str
    tools_used: list[str]
    hypothesis_type: str
    outcome: str  # "supported", "refuted", "inconclusive"
    confidence: float
    cycle_count: int
    wall_time_s: float
    tool_sequence: list[str]
    created_at: str = field(default_factory=_utc_now)


@dataclass
class ToolRecommendation:
    tool_name: str
    success_rate: float
    times_used: int
    recency_weighted_score: float


@dataclass
class StrategyRecommendation:
    hypothesis_types: list[str]
    tool_sequence: list[str]
    expected_cycles: float
    expected_confidence: float
    rationale: str


class MetaLearner:
    """Track and adapt research strategies based on past performance."""

    def __init__(self, store_path: Path | str) -> None:
        self._store_path = Path(store_path)
        self._store_path.parent.mkdir(parents=True, exist_ok=True)
        self._records: list[StrategyRecord] = []
        self._load()

    def _load(self) -> None:
        if self._store_path.exists():
            raw = json.loads(self._store_path.read_text())
            self._records = [StrategyRecord(**r) for r in raw.get("records", [])]

    def _save(self) -> None:
        payload = {"records": [asdict(r) for r in self._records]}
        self._store_path.write_text(json.dumps(payload, indent=2))

    def record_outcome(
        self,
        session_state: Any,
        experiment_record: Any,
        analysis: Any,
    ) -> None:
        """Called after each cycle to record what happened.

        Parameters
        ----------
        session_state : SessionState or compatible mapping
            Must expose ``session_id``, ``cycle``, ``research_question``.
        experiment_record : ExperimentRecord or compatible mapping
            Must expose ``plan`` (with ``"tools"`` key), ``hypothesis_id``.
        analysis : AnalysisResult or compatible mapping
            Must expose ``verdict``, ``confidence`` or ``posterior``.
        """
        plan = (
            experiment_record.plan
            if hasattr(experiment_record, "plan")
            else experiment_record.get("plan", {})
        )
        tools_used: list[str] = []
        tool_sequence: list[str] = []
        if isinstance(plan, dict):
            for step in plan.get("steps", []):
                name = step.get("tool", step.get("tool_name", ""))
                if name:
                    tool_sequence.append(name)
                    if name not in tools_used:
                        tools_used.append(name)

        hypothesis_type = self._infer_hypothesis_type(plan)

        verdict = (
            analysis.verdict
            if hasattr(analysis, "verdict")
            else analysis.get("verdict", "inconclusive")
        )
        confidence = float(
            getattr(analysis, "posterior", None)
            or getattr(analysis, "confidence", None)
            or (analysis.get("posterior") if isinstance(analysis, dict) else None)
            or (analysis.get("confidence") if isinstance(analysis, dict) else None)
            or 0.0
        )

        rq = (
            session_state.research_question
            if hasattr(session_state, "research_question")
            else session_state.get("research_question", "")
        )
        cycle_count = (
            session_state.cycle
            if hasattr(session_state, "cycle")
            else session_state.get("cycle", 1)
        )

        record = StrategyRecord(
            strategy_id=f"strat_{uuid.uuid4().hex[:10]}",
            research_question=rq,
            tools_used=tools_used,
            hypothesis_type=hypothesis_type,
            outcome=verdict,
            confidence=confidence,
            cycle_count=int(cycle_count),
            wall_time_s=0.0,
            tool_sequence=tool_sequence,
        )
        self._records.append(record)
        self._save()

    @staticmethod
    def _infer_hypothesis_type(plan: Any) -> str:
        """Best-effort extraction of hypothesis type from a plan dict."""
        if isinstance(plan, dict):
            for key in ("hypothesis_type", "type", "category"):
                if key in plan:
                    return str(plan[key])
            analysis = plan.get("analysis_type", "")
            if analysis:
                return str(analysis)
        return "unknown"

    def recommend_tools(
        self,
        hypothesis_type: str,
        top_k: int = 5,
    ) -> list[ToolRecommendation]:
        """Recommend tools most likely to produce supported hypotheses.

        Computes per-tool success rate for the given *hypothesis_type*,
        weighted by recency so recent experience counts more.
        """
        relevant = [
            r for r in self._records if r.hypothesis_type == hypothesis_type
        ]
        if not relevant:
            relevant = list(self._records)
        if not relevant:
            return []

        relevant.sort(key=lambda r: r.created_at)
        n = len(relevant)

        tool_successes: dict[str, float] = defaultdict(float)
        tool_totals: dict[str, float] = defaultdict(float)
        tool_counts: dict[str, int] = defaultdict(int)

        for idx, rec in enumerate(relevant):
            recency_weight = 1.0 + idx / max(n, 1)
            is_success = 1.0 if rec.outcome == "supported" else 0.0
            for tool in rec.tools_used:
                tool_successes[tool] += is_success * recency_weight
                tool_totals[tool] += recency_weight
                tool_counts[tool] += 1

        recs: list[ToolRecommendation] = []
        for tool in tool_totals:
            rate = tool_successes[tool] / tool_totals[tool]
            recs.append(
                ToolRecommendation(
                    tool_name=tool,
                    success_rate=round(rate, 4),
                    times_used=tool_counts[tool],
                    recency_weighted_score=round(
                        tool_successes[tool] / max(tool_totals[tool], 1e-9), 4
                    ),
                )
            )

        recs.sort(key=lambda r: r.recency_weighted_score, reverse=True)
        return recs[:top_k]

    def recommend_strategy(
        self,
        research_question: str,
    ) -> StrategyRecommendation:
        """Recommend an overall strategy based on similar past questions.

        Uses keyword overlap to find the most relevant past records, then
        aggregates their hypothesis types, tool sequences, and cycle counts.
        """
        if not self._records:
            return StrategyRecommendation(
                hypothesis_types=[],
                tool_sequence=[],
                expected_cycles=1.0,
                expected_confidence=0.5,
                rationale="No prior experience available.",
            )

        scored = [
            (self._question_similarity(research_question, r.research_question), r)
            for r in self._records
        ]
        scored.sort(key=lambda t: t[0], reverse=True)
        similar = [r for sim, r in scored[:20] if sim > 0]
        if not similar:
            similar = list(self._records)

        supported = [r for r in similar if r.outcome == "supported"]
        pool = supported if supported else similar

        type_counts: dict[str, int] = defaultdict(int)
        for r in pool:
            type_counts[r.hypothesis_type] += 1
        best_types = sorted(type_counts, key=type_counts.get, reverse=True)[:3]  # type: ignore[arg-type]

        seq_counts: dict[tuple[str, ...], int] = defaultdict(int)
        for r in pool:
            seq_counts[tuple(r.tool_sequence)] += 1
        best_seq_tuple = max(seq_counts, key=seq_counts.get) if seq_counts else ()  # type: ignore[arg-type]

        avg_cycles = sum(r.cycle_count for r in pool) / max(len(pool), 1)
        avg_conf = sum(r.confidence for r in pool) / max(len(pool), 1)

        return StrategyRecommendation(
            hypothesis_types=best_types,
            tool_sequence=list(best_seq_tuple),
            expected_cycles=round(avg_cycles, 1),
            expected_confidence=round(avg_conf, 3),
            rationale=(
                f"Based on {len(pool)} similar past outcomes "
                f"({len(supported)} supported). "
                f"Best hypothesis types: {best_types}."
            ),
        )

    @staticmethod
    def _question_similarity(q1: str, q2: str) -> float:
        """Jaccard similarity over lowercased word tokens."""
        a = set(q1.lower().split())
        b = set(q2.lower().split())
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def get_tool_success_rates(self) -> dict[str, float]:
        """Per-tool success rate across all recorded outcomes."""
        tool_total: dict[str, int] = defaultdict(int)
        tool_success: dict[str, int] = defaultdict(int)
        for r in self._records:
            for tool in r.tools_used:
                tool_total[tool] += 1
                if r.outcome == "supported":
                    tool_success[tool] += 1
        return {
            tool: round(tool_success[tool] / total, 4)
            for tool, total in tool_total.items()
        }

    def get_hypothesis_type_stats(self) -> dict[str, dict[str, Any]]:
        """Success/refuted/inconclusive counts per hypothesis type."""
        stats: dict[str, dict[str, int]] = defaultdict(
            lambda: {"supported": 0, "refuted": 0, "inconclusive": 0, "total": 0}
        )
        for r in self._records:
            bucket = stats[r.hypothesis_type]
            bucket["total"] += 1
            if r.outcome in bucket:
                bucket[r.outcome] += 1
        return dict(stats)

    def suggest_prompt_refinements(self) -> list[str]:
        """Analyze patterns in failures and suggest prompt adjustments."""
        suggestions: list[str] = []
        type_stats = self.get_hypothesis_type_stats()

        for htype, counts in type_stats.items():
            if counts["total"] < 2:
                continue
            fail_rate = counts["refuted"] / counts["total"]
            inc_rate = counts["inconclusive"] / counts["total"]

            if fail_rate > 0.6:
                suggestions.append(
                    f"Hypothesis type '{htype}' has a {fail_rate:.0%} refutation rate. "
                    f"Consider more specific, testable predictions for this type, "
                    f"or gather more supporting literature before testing."
                )
            if inc_rate > 0.5:
                suggestions.append(
                    f"Hypothesis type '{htype}' yields {inc_rate:.0%} inconclusive results. "
                    f"The experimental design may need stronger statistical power or "
                    f"clearer success criteria."
                )

        tool_rates = self.get_tool_success_rates()
        for tool, rate in tool_rates.items():
            if rate < 0.2 and sum(
                1 for r in self._records if tool in r.tools_used
            ) >= 3:
                suggestions.append(
                    f"Tool '{tool}' has a low success rate ({rate:.0%}). "
                    f"Consider using alternative tools or adjusting parameters."
                )

        if not suggestions:
            suggestions.append(
                "No clear failure patterns detected yet. Continue accumulating data."
            )
        return suggestions

    def adaptation_summary(self) -> str:
        """Human-readable summary of what the agent has learned."""
        if not self._records:
            return "No strategy data recorded yet."

        total = len(self._records)
        supported = sum(1 for r in self._records if r.outcome == "supported")
        refuted = sum(1 for r in self._records if r.outcome == "refuted")
        inc = total - supported - refuted

        lines = [
            f"Meta-learning summary ({total} recorded strategies):",
            f"  Outcomes: {supported} supported, {refuted} refuted, {inc} inconclusive "
            f"({supported / total:.0%} success rate)",
        ]

        type_stats = self.get_hypothesis_type_stats()
        if type_stats:
            best_type = max(
                type_stats.items(),
                key=lambda kv: kv[1]["supported"] / max(kv[1]["total"], 1),
            )
            lines.append(f"  Best hypothesis type: '{best_type[0]}' "
                         f"({best_type[1]['supported']}/{best_type[1]['total']} supported)")

        tool_rates = self.get_tool_success_rates()
        if tool_rates:
            best_tool = max(tool_rates.items(), key=lambda kv: kv[1])
            lines.append(f"  Best tool: '{best_tool[0]}' ({best_tool[1]:.0%} success rate)")

        refinements = self.suggest_prompt_refinements()
        if refinements and "No clear failure" not in refinements[0]:
            lines.append("  Suggested refinements:")
            for r in refinements[:3]:
                lines.append(f"    - {r}")

        return "\n".join(lines)
