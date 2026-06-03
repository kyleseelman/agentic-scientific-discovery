from __future__ import annotations

import json
from typing import Any

from src.agent.schemas import Hypothesis
from src.config import LLMBackend
from src.memory.knowledge_store import KnowledgeStore, new_id
from src.memory.paper_store import PaperStore
from src.memory.retriever import MemoryRetriever
from src.utils.json_extract import extract_json_object


def _parse_hypotheses(raw: dict[str, Any]) -> list[Hypothesis]:
    hyps = raw.get("hypotheses", raw)
    if isinstance(hyps, dict):
        hyps = [hyps]
    out: list[Hypothesis] = []
    for h in hyps:
        out.append(
            Hypothesis(
                id=str(h.get("id", "")) or "",
                statement=str(h.get("statement", "")),
                rationale=str(h.get("rationale", "")),
                testable_prediction=str(h.get("testable_prediction", "")),
                required_data=[str(x) for x in h.get("required_data", [])],
                confidence_prior=float(h.get("confidence_prior", 0.5)),
                novelty_score=float(h.get("novelty_score", 0.5)),
                status="proposed",
                literature_grounded=bool(h.get("literature_grounded", False)),
                source_papers=[str(x) for x in h.get("source_papers", [])],
            )
        )
    return out


def generate_hypotheses(
    llm: LLMBackend,
    research_question: str,
    dataset_summary: dict[str, Any],
    knowledge: KnowledgeStore,
    retriever: MemoryRetriever,
    available_data_keys: list[str],
    paper_store: PaperStore | None = None,
    suggested_focus: str | None = None,
) -> list[Hypothesis]:
    memory_hits = retriever.retrieve_for_hypothesis(
        research_question + " " + json.dumps(dataset_summary)[:500], k=4
    )
    memory_text = "\n".join(f"- ({h.kind}) {h.text[:240]}" for h in memory_hits)

    literature_context = ""
    if paper_store is not None:
        lit_summary = paper_store.summary_for_context(max_papers=5)
        suggested = paper_store.get_suggested_hypotheses()
        if lit_summary.strip() and lit_summary != "(no papers read yet)":
            literature_context = f"""
Recent literature insights:
{lit_summary}

Previously suggested hypotheses from literature:
{chr(10).join(f'- {s}' for s in suggested[:5]) if suggested else '(none yet)'}
"""

    focus_context = ""
    if suggested_focus:
        focus_context = f"""
Strategic focus directive (from previous strategy assessment):
Prioritize hypotheses related to: {suggested_focus}
"""

    prompt = f"""
You propose testable biological hypotheses for computational validation.

Research question: {research_question}

Dataset summary (JSON): {json.dumps(dataset_summary)[:4000]}

Available data artifacts you may require (choose subset only from this list):
{available_data_keys}

Related prior memory snippets:
{memory_text}
{literature_context}{focus_context}
Return JSON:
{{
  "hypotheses": [
    {{
      "statement": "...",
      "rationale": "...",
      "testable_prediction": "...",
      "required_data": ["..."],
      "confidence_prior": 0.0-1.0,
      "novelty_score": 0.0-1.0,
      "literature_grounded": true/false,
      "source_papers": ["paper title or ID if grounded in literature"]
    }}
  ]
}}

Constraints:
- At least 2 hypotheses; at most 5.
- Each hypothesis must be testable using differential expression, pathway enrichment, and/or public database tools.
- If literature insights are available, at least ONE hypothesis should be grounded in a paper finding (set literature_grounded=true and cite the source).
- For literature-grounded hypotheses, explain in the rationale how the paper's findings motivate this hypothesis.
- Prefer hypotheses that extend, replicate, or challenge findings from the literature.
"""
    text = llm.generate(
        prompt,
        system="You output strictly valid JSON matching the schema.",
        temperature=0.55,
    )
    raw = extract_json_object(text)
    hyps = _parse_hypotheses(raw)
    for h in hyps:
        if not h.id:
            h.id = new_id("hyp")
    return hyps
