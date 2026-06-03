from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.tools.literature import PaperInsights, PaperRecord


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ReadPaper:
    """A paper the agent has read, with extracted insights and provenance."""

    paper: PaperRecord
    insights: PaperInsights
    read_at: str = field(default_factory=_utc_now)
    triggered_hypotheses: list[str] = field(default_factory=list)
    research_context: str = ""


class PaperStore:
    """Persistent store for papers the agent has read and their extracted insights."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._papers: dict[str, ReadPaper] = {}
        self._load()

    @property
    def _path(self) -> Path:
        return self.root / "papers_read.json"

    def _load(self) -> None:
        if not self._path.exists():
            return
        raw = json.loads(self._path.read_text())
        for pid, entry in raw.items():
            paper = PaperRecord(**entry["paper"])
            insights = PaperInsights(**entry["insights"])
            self._papers[pid] = ReadPaper(
                paper=paper,
                insights=insights,
                read_at=entry.get("read_at", ""),
                triggered_hypotheses=entry.get("triggered_hypotheses", []),
                research_context=entry.get("research_context", ""),
            )

    def save(self) -> None:
        data = {}
        for pid, rp in self._papers.items():
            data[pid] = {
                "paper": asdict(rp.paper),
                "insights": asdict(rp.insights),
                "read_at": rp.read_at,
                "triggered_hypotheses": rp.triggered_hypotheses,
                "research_context": rp.research_context,
            }
        self._path.write_text(json.dumps(data, indent=2))

    def add_paper(
        self,
        paper: PaperRecord,
        insights: PaperInsights,
        research_context: str = "",
    ) -> ReadPaper:
        rp = ReadPaper(
            paper=paper,
            insights=insights,
            research_context=research_context,
        )
        self._papers[paper.paper_id] = rp
        return rp

    def link_hypothesis(self, paper_id: str, hypothesis_id: str) -> None:
        if paper_id in self._papers:
            self._papers[paper_id].triggered_hypotheses.append(hypothesis_id)

    def already_read(self, paper_id: str) -> bool:
        return paper_id in self._papers

    def all_papers(self) -> list[ReadPaper]:
        return list(self._papers.values())

    def recent_papers(self, n: int = 10) -> list[ReadPaper]:
        return sorted(self._papers.values(), key=lambda p: p.read_at, reverse=True)[:n]

    def get_all_findings(self) -> list[str]:
        """All key findings across all papers read."""
        return [f for rp in self._papers.values() for f in rp.insights.key_findings]

    def get_all_methods(self) -> list[str]:
        """All unique methods encountered across papers."""
        return list({m for rp in self._papers.values() for m in rp.insights.key_methods})

    def get_suggested_hypotheses(self) -> list[str]:
        """All hypothesis suggestions derived from papers."""
        return [h for rp in self._papers.values() for h in rp.insights.suggested_hypotheses]

    def summary_for_context(self, max_papers: int = 5) -> str:
        """Produce a text summary of recent paper insights for LLM context."""
        lines: list[str] = []
        for rp in self.recent_papers(max_papers):
            lines.append(f"Paper: {rp.paper.title} ({rp.paper.journal}, {rp.paper.date})")
            if rp.insights.key_findings:
                lines.append(f"  Findings: {'; '.join(rp.insights.key_findings[:3])}")
            if rp.insights.key_methods:
                lines.append(f"  Methods: {'; '.join(rp.insights.key_methods[:3])}")
            if rp.insights.suggested_hypotheses:
                lines.append(f"  Suggested: {'; '.join(rp.insights.suggested_hypotheses[:2])}")
            lines.append("")
        return "\n".join(lines) if lines else "(no papers read yet)"
