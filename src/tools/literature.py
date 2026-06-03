from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import urlencode

import requests

from src.config import AppConfig, LLMBackend, get_config
from src.utils.json_extract import extract_json_object


@dataclass
class PaperRecord:
    source: str
    paper_id: str
    title: str
    authors: list[str]
    abstract: str
    journal: str
    date: str
    doi: str = ""
    url: str = ""


@dataclass
class PaperInsights:
    paper_id: str
    title: str
    key_methods: list[str]
    key_findings: list[str]
    limitations: list[str]
    future_work: list[str]
    relevance_summary: str
    suggested_hypotheses: list[str]


# ---------------------------------------------------------------------------
# PubMed: search + fetch full abstracts via NCBI E-utilities
# ---------------------------------------------------------------------------

_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


def search_pubmed(
    query: str, config: AppConfig | None = None, retmax: int = 5
) -> dict[str, Any]:
    cfg = config or get_config()
    try:
        es = requests.get(
            f"{_EUTILS}/esearch.fcgi?db=pubmed&retmode=json&retmax={retmax}&"
            + urlencode({"term": query}),
            timeout=cfg.request_timeout_s,
        )
        es.raise_for_status()
        ids = es.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return {"query": query, "ids": [], "summaries": []}
        time.sleep(0.34)
        esum = requests.get(
            f"{_EUTILS}/esummary.fcgi?db=pubmed&retmode=json&id={','.join(ids)}",
            timeout=cfg.request_timeout_s,
        )
        esum.raise_for_status()
        result = esum.json().get("result", {})
        summaries = []
        for pid in ids:
            rec = result.get(pid, {})
            summaries.append(
                {
                    "id": pid,
                    "title": rec.get("title", ""),
                    "source": rec.get("source", ""),
                    "pubdate": rec.get("pubdate", ""),
                }
            )
        return {"query": query, "ids": ids, "summaries": summaries}
    except requests.RequestException as e:
        return {"query": query, "error": str(e), "ids": [], "summaries": []}


def fetch_pubmed_abstracts(
    pmids: list[str], config: AppConfig | None = None
) -> list[PaperRecord]:
    """Fetch full abstracts for a list of PubMed IDs via efetch XML."""
    cfg = config or get_config()
    if not pmids:
        return []
    try:
        resp = requests.get(
            f"{_EUTILS}/efetch.fcgi",
            params={"db": "pubmed", "id": ",".join(pmids[:20]), "rettype": "xml"},
            timeout=cfg.request_timeout_s,
        )
        resp.raise_for_status()
    except requests.RequestException:
        return []

    papers: list[PaperRecord] = []
    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError:
        return []

    for article_el in root.findall(".//PubmedArticle"):
        pmid_el = article_el.find(".//PMID")
        pmid = pmid_el.text if pmid_el is not None else ""

        title_el = article_el.find(".//ArticleTitle")
        title = title_el.text or "" if title_el is not None else ""

        abstract_parts: list[str] = []
        for at in article_el.findall(".//AbstractText"):
            label = at.get("Label", "")
            text = "".join(at.itertext()).strip()
            if label:
                abstract_parts.append(f"{label}: {text}")
            else:
                abstract_parts.append(text)
        abstract = " ".join(abstract_parts)

        authors: list[str] = []
        for au in article_el.findall(".//Author"):
            last = au.findtext("LastName", "")
            first = au.findtext("Initials", "")
            if last:
                authors.append(f"{last} {first}".strip())

        journal_el = article_el.find(".//Journal/Title")
        journal = journal_el.text or "" if journal_el is not None else ""

        date_el = article_el.find(".//ArticleDate")
        if date_el is not None:
            y = date_el.findtext("Year", "")
            m = date_el.findtext("Month", "")
            date_str = f"{y}-{m}" if m else y
        else:
            y_el = article_el.find(".//PubDate/Year")
            date_str = y_el.text if y_el is not None else ""

        doi = ""
        for eid in article_el.findall(".//ArticleId"):
            if eid.get("IdType") == "doi":
                doi = eid.text or ""
                break

        papers.append(
            PaperRecord(
                source="pubmed",
                paper_id=pmid,
                title=title,
                authors=authors[:10],
                abstract=abstract,
                journal=journal,
                date=date_str,
                doi=doi,
                url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            )
        )
    return papers


# ---------------------------------------------------------------------------
# bioRxiv / medRxiv search via their content API
# ---------------------------------------------------------------------------

def search_biorxiv(
    query: str, server: str = "biorxiv", max_results: int = 5, config: AppConfig | None = None
) -> list[PaperRecord]:
    """Search recent bioRxiv/medRxiv preprints via the content detail API."""
    cfg = config or get_config()
    url = f"https://api.biorxiv.org/details/{server}/2024-01-01/2026-12-31/0/50"
    try:
        resp = requests.get(url, timeout=cfg.request_timeout_s)
        resp.raise_for_status()
        data = resp.json().get("collection", [])
    except requests.RequestException:
        return []

    query_lower = query.lower()
    query_terms = query_lower.split()
    scored: list[tuple[int, dict]] = []
    for item in data:
        text = (item.get("title", "") + " " + item.get("abstract", "")).lower()
        score = sum(1 for t in query_terms if t in text)
        if score > 0:
            scored.append((score, item))

    scored.sort(key=lambda x: -x[0])
    papers: list[PaperRecord] = []
    for _, item in scored[:max_results]:
        papers.append(
            PaperRecord(
                source=server,
                paper_id=item.get("doi", ""),
                title=item.get("title", ""),
                authors=[a.strip() for a in item.get("authors", "").split(";") if a.strip()][:10],
                abstract=item.get("abstract", ""),
                journal=server,
                date=item.get("date", ""),
                doi=item.get("doi", ""),
                url=f"https://doi.org/{item.get('doi', '')}",
            )
        )
    return papers


# ---------------------------------------------------------------------------
# LLM-powered paper insight extraction
# ---------------------------------------------------------------------------

def _is_mock_llm(llm: LLMBackend) -> bool:
    """Detect if the LLM is a MockLLMBackend to avoid generating misleading insights."""
    return type(llm).__name__ == "MockLLMBackend"


def extract_paper_insights(
    paper: PaperRecord,
    research_context: str,
    llm: LLMBackend,
) -> PaperInsights:
    """Use an LLM to extract structured insights from a paper abstract."""
    if _is_mock_llm(llm):
        abstract = paper.abstract or ""
        return PaperInsights(
            paper_id=paper.paper_id,
            title=paper.title,
            key_methods=[],
            key_findings=[abstract[:200]] if abstract else [],
            limitations=[],
            future_work=[],
            relevance_summary=f"Abstract: {abstract[:300]}",
            suggested_hypotheses=[],
        )

    prompt = f"""Extract structured scientific insights from this paper abstract.

Paper title: {paper.title}
Authors: {', '.join(paper.authors[:5])}
Journal: {paper.journal} ({paper.date})
Abstract: {paper.abstract[:3000]}

Current research context: {research_context[:500]}

Return JSON:
{{
  "key_methods": ["method1", "method2"],
  "key_findings": ["finding1", "finding2"],
  "limitations": ["limitation1"],
  "future_work": ["direction1"],
  "relevance_summary": "How this paper relates to our current research...",
  "suggested_hypotheses": ["hypothesis we could test based on this paper..."]
}}

Focus on extractable, testable scientific claims. Be specific about methods and quantitative findings."""

    text = llm.generate(prompt, system="Return strictly valid JSON.", temperature=0.3)
    obj = extract_json_object(text)
    return PaperInsights(
        paper_id=paper.paper_id,
        title=paper.title,
        key_methods=[str(x) for x in obj.get("key_methods", [])],
        key_findings=[str(x) for x in obj.get("key_findings", [])],
        limitations=[str(x) for x in obj.get("limitations", [])],
        future_work=[str(x) for x in obj.get("future_work", [])],
        relevance_summary=str(obj.get("relevance_summary", "")),
        suggested_hypotheses=[str(x) for x in obj.get("suggested_hypotheses", [])],
    )


# ---------------------------------------------------------------------------
# High-level: literature scan → papers → insights pipeline
# ---------------------------------------------------------------------------

def literature_scan(
    query: str,
    research_context: str,
    llm: LLMBackend,
    config: AppConfig | None = None,
    max_papers: int = 3,
    include_preprints: bool = True,
) -> dict[str, Any]:
    """Full pipeline: search PubMed (+ optionally bioRxiv), fetch abstracts, extract insights."""
    cfg = config or get_config()

    search_result = search_pubmed(query, config=cfg, retmax=max_papers)
    pmids = search_result.get("ids", [])
    papers = fetch_pubmed_abstracts(pmids, config=cfg)

    if include_preprints:
        preprints = search_biorxiv(query, config=cfg, max_results=max(1, max_papers // 2))
        papers.extend(preprints)

    insights: list[PaperInsights] = []
    for paper in papers[:max_papers]:
        if not paper.abstract:
            continue
        ins = extract_paper_insights(paper, research_context, llm)
        insights.append(ins)

    return {
        "query": query,
        "papers_found": len(papers),
        "papers_analyzed": len(insights),
        "papers": [asdict(p) for p in papers],
        "insights": [asdict(i) for i in insights],
        "all_suggested_hypotheses": [
            h for ins in insights for h in ins.suggested_hypotheses
        ],
        "all_methods_seen": list(
            {m for ins in insights for m in ins.key_methods}
        ),
    }


def literature_context_for_genes(
    genes: list[str], config: AppConfig | None = None
) -> dict[str, Any]:
    parts = []
    for g in genes[:5]:
        q = f"{g}[Title/Abstract] AND (gene expression OR pathway)"
        parts.append(search_pubmed(q, config=config, retmax=2))
    return {"gene_queries": parts}
