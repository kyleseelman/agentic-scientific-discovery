"""Autonomous research question discovery from literature.

Given a broad topic (e.g. "Alzheimer's disease"), the agent:
1. Searches PubMed + bioRxiv for recent papers
2. Extracts key findings, gaps, and future directions
3. Uses the LLM to synthesize novel research questions
4. Searches GEO for relevant datasets
5. Returns ranked (question, dataset) pairs ready for investigation
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

import requests

from src.config import AppConfig, LLMBackend, get_config
from src.tools.literature import (
    PaperRecord,
    extract_paper_insights,
    fetch_pubmed_abstracts,
    search_biorxiv,
    search_pubmed,
)
from src.utils.json_extract import extract_json_object


@dataclass
class ResearchProposal:
    question: str
    rationale: str
    source_papers: list[str]
    suggested_geo: str
    geo_rationale: str
    novelty: str
    feasibility: str


def search_geo_datasets(
    query: str, max_results: int = 5, config: AppConfig | None = None,
) -> list[dict[str, Any]]:
    """Search NCBI GEO for expression datasets matching a query."""
    cfg = config or get_config()
    eutils = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    geo_query = f"{query} AND gse[ETYP] AND Homo sapiens[ORGN]"

    try:
        resp = requests.get(
            f"{eutils}/esearch.fcgi",
            params={"db": "gds", "retmode": "json", "retmax": max_results,
                    "term": geo_query, "sort": "relevance"},
            timeout=cfg.request_timeout_s,
        )
        resp.raise_for_status()
        ids = resp.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []

        time.sleep(0.34)
        summary_resp = requests.get(
            f"{eutils}/esummary.fcgi",
            params={"db": "gds", "retmode": "json", "id": ",".join(ids)},
            timeout=cfg.request_timeout_s,
        )
        summary_resp.raise_for_status()
        result = summary_resp.json().get("result", {})

        datasets = []
        for gid in ids:
            rec = result.get(gid, {})
            accession = rec.get("accession", "")
            if not accession.startswith("GSE"):
                continue
            datasets.append({
                "accession": accession,
                "title": rec.get("title", ""),
                "summary": rec.get("summary", "")[:300],
                "n_samples": rec.get("n_samples", 0),
                "taxon": rec.get("taxon", ""),
                "gpl": rec.get("gpl", ""),
                "gdstype": rec.get("gdstype", ""),
            })
        return datasets

    except requests.RequestException:
        return []


def discover_research_questions(
    topic: str,
    llm: LLMBackend,
    config: AppConfig | None = None,
    max_papers: int = 8,
    max_proposals: int = 3,
) -> list[ResearchProposal]:
    """Autonomously discover novel research questions from literature.

    1. Search PubMed + bioRxiv for recent papers on the topic
    2. Extract insights (findings, gaps, future work)
    3. Search GEO for relevant datasets
    4. Use LLM to synthesize novel, testable research questions
    """
    cfg = config or get_config()

    # --- Step 1: Literature scan ---
    print(f"  [Discovery] Searching literature for: {topic}")
    pubmed_result = search_pubmed(
        f"{topic} AND (gene expression OR transcriptomics OR RNA-seq)",
        config=cfg, retmax=max_papers,
    )
    pmids = pubmed_result.get("ids", [])
    papers = fetch_pubmed_abstracts(pmids, config=cfg)
    preprints = search_biorxiv(topic, config=cfg, max_results=3)
    papers.extend(preprints)
    print(f"  [Discovery] Found {len(papers)} papers")

    # --- Step 2: Extract insights ---
    all_findings: list[str] = []
    all_gaps: list[str] = []
    all_future: list[str] = []
    paper_summaries: list[str] = []

    for paper in papers[:max_papers]:
        if not paper.abstract:
            continue
        insights = extract_paper_insights(paper, topic, llm)
        all_findings.extend(insights.key_findings)
        all_gaps.extend(insights.limitations)
        all_future.extend(insights.future_work)
        paper_summaries.append(
            f"- [{paper.paper_id}] {paper.title}: "
            f"{insights.relevance_summary[:200]}"
        )

    print(f"  [Discovery] Extracted {len(all_findings)} findings, "
          f"{len(all_gaps)} limitations, {len(all_future)} future directions")

    # --- Step 3: Search GEO for datasets ---
    print(f"  [Discovery] Searching GEO for datasets on: {topic}")
    geo_datasets = search_geo_datasets(topic, max_results=10, config=cfg)
    geo_text = "\n".join(
        f"- {d['accession']}: {d['title']} ({d['n_samples']} samples)"
        for d in geo_datasets
    ) if geo_datasets else "(no GEO datasets found — user will need to specify one)"

    print(f"  [Discovery] Found {len(geo_datasets)} GEO datasets")

    # --- Step 4: LLM synthesizes research questions ---
    prompt = f"""You are a computational biology researcher. Based on a literature review,
propose {max_proposals} novel, testable research questions.

TOPIC: {topic}

RECENT PAPERS REVIEWED:
{chr(10).join(paper_summaries[:10])}

KEY FINDINGS IN THE FIELD:
{chr(10).join(f'- {f}' for f in all_findings[:15])}

GAPS AND LIMITATIONS IDENTIFIED:
{chr(10).join(f'- {g}' for g in all_gaps[:10])}

FUTURE DIRECTIONS SUGGESTED:
{chr(10).join(f'- {f}' for f in all_future[:10])}

AVAILABLE GEO DATASETS:
{geo_text}

Return JSON:
{{
  "proposals": [
    {{
      "question": "A specific, testable research question...",
      "rationale": "Why this is novel and important, citing specific papers...",
      "source_papers": ["paper_id1", "paper_id2"],
      "suggested_geo": "GSExxxxx",
      "geo_rationale": "Why this dataset is appropriate for testing this question...",
      "novelty": "What makes this question novel compared to existing work...",
      "feasibility": "Why this can be answered with available computational tools..."
    }}
  ]
}}

Requirements:
- Each question must be answerable using gene expression data + computational tools
- Questions should address GAPS in the literature, not just replicate existing findings
- Prefer questions that connect two concepts in a novel way
- Each question should specify a real GEO dataset from the list above
- If no dataset from the list fits, suggest what type of dataset would be needed
- Rank by novelty × feasibility
"""

    print(f"  [Discovery] Generating research proposals with LLM...")
    text = llm.generate(
        prompt,
        system="You are a senior computational biologist proposing novel research. "
               "Output strictly valid JSON.",
        temperature=0.6,
    )
    obj = extract_json_object(text)

    proposals = []
    for p in obj.get("proposals", []):
        proposals.append(ResearchProposal(
            question=str(p.get("question", "")),
            rationale=str(p.get("rationale", "")),
            source_papers=[str(s) for s in p.get("source_papers", [])],
            suggested_geo=str(p.get("suggested_geo", "")),
            geo_rationale=str(p.get("geo_rationale", "")),
            novelty=str(p.get("novelty", "")),
            feasibility=str(p.get("feasibility", "")),
        ))

    return proposals
